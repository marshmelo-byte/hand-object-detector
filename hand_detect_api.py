from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import _init_paths
from model.utils.config import cfg, cfg_from_list
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
import torch

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()
print(f"Hand detector using device: {DEVICE}")

fasterRCNN = None


@app.on_event("startup")
def load_model():
    global fasterRCNN

    cfg_from_list(["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"])
    cfg.USE_GPU_NMS = USE_GPU
    cfg.CUDA = USE_GPU

    # FIX: load checkpoint FIRST so we can read pooling_mode from it
    checkpoint = torch.load(
        "models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth",
        map_location=DEVICE,
    )

    # FIX: set POOLING_MODE BEFORE create_architecture() — this was causing
    # 'pooled_feat referenced before assignment' crash because checkpoint uses
    # 'align' but cfg defaults to 'crop', so the if/elif never matched.
    if "pooling_mode" in checkpoint:
        cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print(f"POOLING_MODE set to: {cfg.POOLING_MODE}")
    else:
        cfg.POOLING_MODE = "align"  # safe default for this model
        print(f"POOLING_MODE not in checkpoint, defaulting to: {cfg.POOLING_MODE}")

    # NOW create architecture with the correct pooling mode
    fasterRCNN = resnet(PASCAL_CLASSES, 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()

    fasterRCNN.load_state_dict(checkpoint["model"])
    fasterRCNN.to(DEVICE).eval()
    print(f"Hand object detector loaded on {DEVICE}!")


def _get_image_blob(im):
    """
    Mirror demo.py's _get_image_blob exactly.
    Keeps BGR order, subtracts cfg.PIXEL_MEANS, builds the multi-scale blob.
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[:2])
    im_size_max = np.max(im_shape[:2])

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


@app.post("/detect_contact")
async def detect_contact(request: dict = Body(...)):
    """
    Input:  {"image": "<base64 or data-uri>", "thresh_hand": 0.5, "thresh_obj": 0.5}
    Output: {
        "hand_detections":   [{"bbox": [x1,y1,x2,y2], "score": float, "contact_state": str, "hand_side": str}, ...],
        "object_detections": [{"bbox": [x1,y1,x2,y2], "score": float}, ...],
        "contact_validated": bool,
        "contact_state_map": {...},
        "num_hands": int,
        "num_objects": int,
    }
    """
    try:
        # ── Decode image ─────────────────────────────────────────────────────────
        img_b64 = request.get("image", "")
        if img_b64.startswith("data:image"):
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # BGR, as the model expects

        thresh_hand = float(request.get("thresh_hand", request.get("threshold", 0.5)))
        thresh_obj  = float(request.get("thresh_obj",  request.get("threshold", 0.5)))

        # ── Preprocess ───────────────────────────────────────────────────────────
        blobs, im_scales = _get_image_blob(frame)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
        )

        im_data = torch.from_numpy(im_blob).permute(0, 3, 1, 2).to(DEVICE)
        im_info = torch.from_numpy(im_info_np).to(DEVICE)
        gt_boxes = torch.zeros(1, 1, 5, device=DEVICE)
        num_boxes = torch.zeros(1, device=DEVICE)
        box_info  = torch.zeros(1, 1, 5, device=DEVICE)

        # ── Inference ────────────────────────────────────────────────────────────
        with torch.no_grad():
            rois, cls_prob, bbox_pred, \
            _rpn_cls, _rpn_box, \
            _rcnn_cls, _rcnn_bbox, \
            _rois_label, loss_list = fasterRCNN(
                im_data, im_info, gt_boxes, num_boxes, box_info
            )

        # ── Extract contact state and hand side from loss_list ───────────────────
        contact_vector = loss_list[0][0]            # [1, N, 5]
        lr_vector      = loss_list[2][0].detach()   # [1, N, 1]

        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()  # [N, 1]
        lr_pred = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()      # [N, 1]

        # ── Bounding-box regression ──────────────────────────────────────────────
        scores = cls_prob.data
        boxes  = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                stds  = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(DEVICE)
                means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(DEVICE)
                box_deltas = box_deltas.view(-1, 4) * stds + means
                box_deltas = box_deltas.view(1, -1, 4 * len(PASCAL_CLASSES))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores     = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        # ── Debug logging ─────────────────────────────────────────────────────────
        print(
            f"[HAND_DETECTOR] hand max_score={float(scores[:, 2].max()):.3f}, "
            f"above_thresh={int((scores[:, 2] > thresh_hand).sum())}"
        )
        print(
            f"[HAND_DETECTOR] obj  max_score={float(scores[:, 1].max()):.3f}, "
            f"above_thresh={int((scores[:, 1] > thresh_obj).sum())}"
        )

        # ── Per-class NMS + build output ──────────────────────────────────────────
        hand_dets = []
        obj_dets  = []

        class_cfg = [
            (2, "hand",   thresh_hand),
            (1, "object", thresh_obj),
        ]

        for cls_idx, label, thresh in class_cfg:
            cls_scores = scores[:, cls_idx]
            inds = torch.nonzero(cls_scores > thresh).view(-1)

            if inds.numel() == 0:
                continue

            cls_scores_kept = cls_scores[inds]
            cls_boxes_kept  = pred_boxes[inds][:, cls_idx * 4 : (cls_idx + 1) * 4]

            _, order = torch.sort(cls_scores_kept, descending=True)
            cls_boxes_ord  = cls_boxes_kept[order]
            cls_scores_ord = cls_scores_kept[order]

            keep = nms(cls_boxes_ord, cls_scores_ord, cfg.TEST.NMS)
            keep = keep.view(-1).long()

            final_boxes  = cls_boxes_ord[keep].cpu().numpy()
            final_scores = cls_scores_ord[keep].cpu().numpy()
            original_inds = inds[order][keep]

            for k in range(len(keep)):
                x1, y1, x2, y2 = final_boxes[k].tolist()
                score = float(final_scores[k])
                orig_idx = int(original_inds[k])

                if label == "hand":
                    c_state = CONTACT_STATES[int(contact_indices[orig_idx].item())]
                    h_side  = HAND_SIDE[int(lr_pred[orig_idx].item())]
                    hand_dets.append({
                        "bbox":          [x1, y1, x2, y2],
                        "score":         score,
                        "contact_state": c_state,
                        "hand_side":     h_side,
                    })
                else:
                    obj_dets.append({
                        "bbox":  [x1, y1, x2, y2],
                        "score": score,
                    })

        print(f"[HAND_DETECTOR] num_hands={len(hand_dets)}, num_objects={len(obj_dets)}")

        # contact_validated = any hand NOT in "No contact" state
        contact_validated = any(
            det["contact_state"] != "N" for det in hand_dets
        )

        return {
            "hand_detections":   hand_dets,
            "object_detections": obj_dets,
            "contact_validated": contact_validated,
            "contact_state_map": CONTACT_STATES,
            "num_hands":         len(hand_dets),
            "num_objects":       len(obj_dets),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "contact_validated": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)