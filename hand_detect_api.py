from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
import _init_paths
import torch

# Set cfg BEFORE any model module imports
from model.utils.config import cfg, cfg_from_list
cfg_from_list(["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"])
USE_GPU = torch.cuda.is_available()
DEVICE  = torch.device("cuda" if USE_GPU else "cpu")
cfg.USE_GPU_NMS = USE_GPU
cfg.CUDA        = USE_GPU

from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Contact state labels (N=No contact, S=Self, O=Other person, P=Portable obj, F=Fixed obj)
CONTACT_STATES = {0: "N", 1: "S", 2: "O", 3: "P", 4: "F"}
HAND_SIDE = {0: "Left", 1: "Right"}

PASCAL_CLASSES = np.asarray(["__background__", "targetobject", "hand"])
print(f"Hand detector using device: {DEVICE}")

fasterRCNN = None

# Pre-allocate tensors exactly like demo.py does
im_data   = torch.FloatTensor(1)
im_info   = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes  = torch.FloatTensor(1)
box_info  = torch.FloatTensor(1)


@app.on_event("startup")
def load_model():
    global fasterRCNN, im_data, im_info, num_boxes, gt_boxes, box_info

    checkpoint = torch.load(
        "models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth",
        map_location=DEVICE,
    )

    if "pooling_mode" in checkpoint:
        cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print(f"POOLING_MODE set to: {cfg.POOLING_MODE}")
    else:
        cfg.POOLING_MODE = "align"
        print(f"POOLING_MODE not in checkpoint, defaulting to: {cfg.POOLING_MODE}")

    fasterRCNN = resnet(PASCAL_CLASSES, 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()
    fasterRCNN.load_state_dict(checkpoint["model"])

    # Ship pre-allocated tensors + model to GPU, exactly like demo.py
    if USE_GPU:
        im_data   = im_data.cuda()
        im_info   = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes  = gt_boxes.cuda()
        fasterRCNN.cuda()

    fasterRCNN.eval()
    print(f"Hand object detector loaded on {DEVICE}!")


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        resized = cv2.resize(
            im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        im_scale_factors.append(im_scale)
        processed_ims.append(resized)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)

def find_matched_object(hand_det, obj_dets):
    """
    Match a hand to its held object using the offset vector,
    exactly like filter_object() in the original demo.py.
    
    Returns the matched object dict, or None if no match found.
    """
    if not obj_dets:
        return None

    # Build object center list
    obj_centers = []
    for obj in obj_dets:
        x1, y1, x2, y2 = obj["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        obj_centers.append((cx, cy))

    # Get hand center
    hx1, hy1, hx2, hy2 = hand_det["bbox"]
    hand_cx = (hx1 + hx2) / 2
    hand_cy = (hy1 + hy2) / 2

    # Compute projected point using offset vector (mirrors demo.py exactly)
    mag, dx, dy = hand_det["offset_vector"]
    proj_x = hand_cx + mag * 10000 * dx
    proj_y = hand_cy + mag * 10000 * dy

    # Find nearest object center to projected point
    min_dist = float("inf")
    best_obj = None
    for i, (cx, cy) in enumerate(obj_centers):
        dist = (cx - proj_x) ** 2 + (cy - proj_y) ** 2
        if dist < min_dist:
            min_dist = dist
            best_obj = obj_dets[i]

    return best_obj


def validate_contact(hand_det, obj_dets, obj_score_thresh=0.95):
    """
    Returns (is_valid, matched_object) where is_valid is True only when:
      1. contact_state == "P"
      2. A matched object exists
      3. That object's score >= obj_score_thresh
    """
    # Condition 1: model must predict portable-object contact
    if hand_det.get("contact_state") != "P":
        return False, None

    # Condition 2 & 3: find matched object and check its score
    matched_obj = find_matched_object(hand_det, obj_dets)
    if matched_obj is None:
        return False, None

    if matched_obj["score"] < obj_score_thresh:
        print(
            f"[CONTACT] Rejected — matched object score {matched_obj['score']:.3f} "
            f"< threshold {obj_score_thresh}"
        )
        return False, None

    return True, matched_obj

@app.post("/detect_contact")
async def detect_contact(request: dict = Body(...)):
    """
    Input:  {"image": "<base64 or data-uri>", "thresh_hand": 0.5, "thresh_obj": 0.5}
    Output: {
        "hand_detections":   [{"bbox": [x1,y1,x2,y2], "score": float, "contact_state": str, "hand_side": str, "offset_vector": [mag, dx, dy]}, ...],
        "object_detections": [{"bbox": [x1,y1,x2,y2], "score": float}, ...],
        "contact_validated": bool,
        "contact_state_map": {...},
        "num_hands": int,
        "num_objects": int,
    }
    offset_vector: [magnitude, unit_dx, unit_dy] — points from hand center toward
    the object it is holding. Use as: projected = hand_center + magnitude * 10000 * [dx, dy]
    """
    try:
        # ── Decode image ──────────────────────────────────────────────────────
        img_b64 = request.get("image", "")
        if img_b64.startswith("data:image"):
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR, as model expects

        thresh_hand = float(request.get("thresh_hand", request.get("threshold", 0.5)))
        thresh_obj  = float(request.get("thresh_obj",  request.get("threshold", 0.5)))

        # ── Preprocess ────────────────────────────────────────────────────────
        blobs, im_scales = _get_image_blob(frame)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob    = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
        )

        im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        # ── Load into pre-allocated tensors exactly like demo.py ──────────────
        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_()

        # ── Inference ─────────────────────────────────────────────────────────
        rois, cls_prob, bbox_pred, \
        _rpn_loss_cls, _rpn_loss_box, \
        _rcnn_loss_cls, _rcnn_loss_bbox, \
        _rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

        # ── Extract contact state, offset vector, and hand side ───────────────
        contact_vector = loss_list[0][0]            # [1, N, 5]
        offset_vector  = loss_list[1][0].detach()   # [1, N, 3] — magnitude, unit_dx, unit_dy
        lr_vector      = loss_list[2][0].detach()   # [1, N, 1]

        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
        offset_squeezed = offset_vector.squeeze(0)  # [N, 3]
        lr_pred = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

        # ── Bounding-box regression ───────────────────────────────────────────
        scores = cls_prob.data
        boxes  = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                if USE_GPU:
                    stds  = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                    means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    stds  = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
                    means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(-1, 4) * stds + means
                box_deltas = box_deltas.view(1, -1, 4 * len(PASCAL_CLASSES))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        # Squeeze exactly like demo.py — resize_/copy_ guarantees contiguous memory
        scores     = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        # ── Debug logging ─────────────────────────────────────────────────────
        print(
            f"[HAND_DETECTOR] hand max_score={float(scores[:, 2].max()):.3f}, "
            f"above_thresh={int((scores[:, 2] > thresh_hand).sum())}"
        )
        print(
            f"[HAND_DETECTOR] obj  max_score={float(scores[:, 1].max()):.3f}, "
            f"above_thresh={int((scores[:, 1] > thresh_obj).sum())}"
        )

        # ── Per-class NMS + build output (mirrors demo.py loop) ───────────────
        hand_dets = []
        obj_dets  = []

        for j in range(1, len(PASCAL_CLASSES)):
            cls_name = PASCAL_CLASSES[j]
            thresh   = thresh_hand if cls_name == "hand" else thresh_obj

            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() == 0:
                continue

            cls_scores = scores[:, j][inds]
            cls_boxes  = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            _, order = torch.sort(cls_scores, 0, True)

            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            keep = keep.view(-1).long()

            final_boxes  = cls_boxes[order][keep].cpu().numpy()
            final_scores = cls_scores[order][keep].cpu().numpy()
            kept_inds    = inds[order][keep]

            for k in range(len(keep)):
                x1, y1, x2, y2 = final_boxes[k].tolist()
                score    = float(final_scores[k])
                orig_idx = int(kept_inds[k])

                if cls_name == "hand":
                    c_state = CONTACT_STATES[int(contact_indices[orig_idx].item())]
                    h_side  = HAND_SIDE[int(lr_pred[orig_idx].item())]
                    ov      = offset_squeezed[orig_idx].cpu().numpy().tolist()  # [mag, dx, dy]
                    hand_dets.append({
                        "bbox":          [x1, y1, x2, y2],
                        "score":         score,
                        "contact_state": c_state,
                        "hand_side":     h_side,
                        "offset_vector": ov,  # [magnitude, unit_dx, unit_dy]
                    })
                else:  # targetobject
                    obj_dets.append({
                        "bbox":  [x1, y1, x2, y2],
                        "score": score,
                    })

        print(f"[HAND_DETECTOR] num_hands={len(hand_dets)}, num_objects={len(obj_dets)}")

       
        contact_validated = False
        contact_matched_object = None

        for hand in hand_dets:
            is_valid, matched_obj = validate_contact(hand, obj_dets, obj_score_thresh=0.90)
            if is_valid:
                contact_validated = True
                contact_matched_object = matched_obj
                break

        print(f"[HAND_DETECTOR] full response: {json.dumps({'hand_detections': hand_dets, 'object_detections': obj_dets, 'contact_validated': contact_validated}, indent=2)}")

        return {
            "hand_detections": hand_dets,
            "object_detections": obj_dets,

            # main reach signal
            "contact_validated": contact_validated,

            # extra debug fields
            "matched_object_present": contact_matched_object is not None,
            "contact_matched_object": contact_matched_object,
            "matched_object_score": (
                contact_matched_object["score"]
                if contact_matched_object is not None
                else None
            ),

            "contact_state_map": CONTACT_STATES,
            "num_hands": len(hand_dets),
            "num_objects": len(obj_dets),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "contact_validated": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)