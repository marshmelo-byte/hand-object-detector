from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import _init_paths
from model.utils.config import cfg
from model.faster_rcnn.resnet import resnet
from model.roi_layers import nms
import torch
import torchvision.transforms as T
from PIL import Image
from io import BytesIO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Contact state labels
CONTACT_STATES = {0: "N", 1: "S", 2: "O", 3: "P", 4: "F"}
HAND_SIDE = {0: "Left", 1: "Right"}

# Load model once at startup
fasterRCNN = None

@app.on_event("startup")
def load_model():
    global fasterRCNN
    cfg.USE_GPU_NMS = True
    cfg.CUDA = True

    fasterRCNN = resnet(["__background__", "targetobject", "hand"], 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()
    
    checkpoint = torch.load(
        "models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth"
    )
    fasterRCNN.load_state_dict(checkpoint["model"])
    fasterRCNN.cuda().eval()
    print("Hand object detector loaded!")


def preprocess(frame):
    """Preprocess frame for the model"""
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im)
    
    # Resize keeping aspect ratio
    target_size = 600
    w, h = im_pil.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    im_pil = im_pil.resize((new_w, new_h))
    
    transform = T.Compose([T.ToTensor()])
    im_tensor = transform(im_pil).unsqueeze(0).cuda()
    im_info = torch.tensor([[new_h, new_w, scale]]).cuda()
    gt_boxes = torch.zeros(1, 1, 5).cuda()
    num_boxes = torch.zeros(1).cuda()
    
    return im_tensor, im_info, gt_boxes, num_boxes, scale


@app.post("/detect_contact")
async def detect_contact(request: dict = Body(...)):
    """
    Input: {"image": "<base64>", "threshold": 0.5}
    Output: {"hand_detections": [...], "object_detections": [...], "contact_validated": bool}
    """
    try:
        # Decode image
        img_b64 = request.get("image", "")
        if img_b64.startswith("data:image"):
            img_b64 = img_b64.split(",", 1)[1]
        
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        thresh = request.get("threshold", 0.5)
        
        # Preprocess
        im_tensor, im_info, gt_boxes, num_boxes, scale = preprocess(frame)
        
        # Inference
        with torch.no_grad():
            rois, cls_prob, bbox_pred, _, _, _, _, _ = fasterRCNN(
                im_tensor, im_info, gt_boxes, num_boxes
            )
        
        # Parse detections (same logic as demo.py)
        scores = cls_prob.data.squeeze()
        boxes = rois.data.squeeze()[:, 1:5]
        
        hand_dets = []
        obj_dets = []
        contact_validated = False
        
        # Class 1 = object, Class 2 = hand
        for cls_idx, label in [(1, "object"), (2, "hand")]:
            cls_scores = scores[:, cls_idx]
            keep = cls_scores > thresh
            cls_boxes = boxes[keep] / scale
            cls_scores_kept = cls_scores[keep]
            
            for i in range(len(cls_scores_kept)):
                x1, y1, x2, y2 = cls_boxes[i].cpu().numpy().tolist()
                score = float(cls_scores_kept[i])
                
                if label == "hand":
                    # hand_dets: [x1,y1,x2,y2, score, contact_state, dx, dy, dz, side]
                    det = [x1, y1, x2, y2, score]
                    hand_dets.append(det)
                else:
                    obj_dets.append([x1, y1, x2, y2, score])
        
        # Validate contact: hand detected with contact state P (portable) or O (other)
        if hand_dets:
            contact_validated = True  # Hand + object both present = contact possible
        
        return {
            "hand_detections": hand_dets,
            "object_detections": obj_dets,
            "contact_validated": contact_validated,
            "contact_state_map": CONTACT_STATES,
            "num_hands": len(hand_dets),
            "num_objects": len(obj_dets)
        }
    
    except Exception as e:
        return {"error": str(e), "contact_validated": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
