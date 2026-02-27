"""
Nerf Target Detection API Server
Runs YOLO inference and returns target coordinates
"""

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load model once at startup
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # Or your custom model
print("✓ Model loaded")

# ============================================================
# SETTINGS YOU MAY WANT TO CHANGE
# ============================================================
CONF_THRESHOLD = 0.3          # YOLO confidence threshold
MIN_AREA = 200                # Minimum detection area in pixels
GREEN_RATIO_MIN = 0.05        # Minimum green content (5%)

# Green color range in HSV
GREEN_LOWER = np.array([35, 60, 40])
GREEN_UPPER = np.array([80, 255, 255])
# ============================================================


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Process frame and return target info"""
    
    # Decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Could not decode image"}
    
    h, w = frame.shape[:2]
    
    # Convert to HSV for green detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    # Run YOLO
    results = model(frame, verbose=False, conf=CONF_THRESHOLD)
    
    # Build response
    response = {
        "found": False,
        "center": None,
        "bbox": None,
        "confidence": 0,
        "area": 0,
        "human_detected": False,
        "frame_center": [w // 2, h // 2]
    }
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        valid_detections = []
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # =============================================
            # HUMAN AVOIDANCE - DO NOT REMOVE
            # =============================================
            if cls == 0:  # Person class
                response["human_detected"] = True
                continue
            # =============================================
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Check green content in box
            roi = green_mask[y1:y2, x1:x2]
            if roi.size > 0:
                green_ratio = np.sum(roi > 0) / roi.size
                
                if green_ratio > GREEN_RATIO_MIN:
                    area = (x2 - x1) * (y2 - y1)
                    if area > MIN_AREA:
                        valid_detections.append({
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "area": int(area),
                            "conf": conf
                        })
        
        # Pick best detection
        if valid_detections:
            best = max(valid_detections, key=lambda x: x["conf"])
            cx = (best["x1"] + best["x2"]) // 2
            cy = (best["y1"] + best["y2"]) // 2
            
            response.update({
                "found": True,
                "center": [cx, cy],
                "bbox": [best["x1"], best["y1"], 
                         best["x2"] - best["x1"], 
                         best["y2"] - best["y1"]],
                "confidence": best["conf"],
                "area": best["area"]
            })
    
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8n"}

