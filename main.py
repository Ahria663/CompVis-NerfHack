import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image

# --- CONFIGURATION & PATHS ---
# TWEAK HERE: Replace "path/to/your/custom_model" with the folder containing 
# your trained 'config.json' and 'model.safetensors' files.
CUSTOM_MODEL_PATH = "./models/nerfhack-detector"

# TWEAK HERE: Replace with the exact label name you used during training (e.g., 'target')
MY_CUSTOM_LABEL = "target" 

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your CUSTOM fine-tuned model
# Note: This replaces the generic 'facebook/detr-resnet-50'
detector = pipeline("object-detection", model=CUSTOM_MODEL_PATH, device=device)

# --- SYSTEM SETTINGS ---
MIN_CONFIDENCE = 0.7  # We can set this higher because the model is specialized
SKIP_FRAMES = 3       # Fewer frames to skip if your laptop has a good GPU
cap = cv2.VideoCapture(0)
frame_count = 0
last_results = []

print(f"System Active. Hunting for custom target: {MY_CUSTOM_LABEL}...")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape

    # Performance optimization
    if frame_count % SKIP_FRAMES == 0:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_img.thumbnail((400, 400)) # Keep resizing for speed
        
        # Run inference using your custom weights
        raw_results = detector(pil_img, threshold=MIN_CONFIDENCE)

        valid_targets = []
        for res in raw_results:
            # Check if the AI found the specific object you trained it on
            if res['label'] == MY_CUSTOM_LABEL:
                # Re-scale coordinates to full frame
                box = res['box']
                scale_x, scale_y = w / 400, h / 400
                res['scaled_box'] = {
                    'xmin': box['xmin'] * scale_x, 'ymin': box['ymin'] * scale_y,
                    'xmax': box['xmax'] * scale_x, 'ymax': box['ymax'] * scale_y
                }
                valid_targets.append(res)

        # SELECTION LOGIC: If multiple targets exist, pick the largest one (likely closest)
        if valid_targets:
            best_target = max(valid_targets, key=lambda x: (x['box']['xmax'] - x['box']['xmin']))
            last_results = [best_target]
        else:
            last_results = []

    # --- UI DRAWING ---
    for res in last_results:
        b = res['scaled_box']
        # Drawing a specialized target lock UI
        cv2.rectangle(frame, (int(b['xmin']), int(b['ymin'])), (int(b['xmax']), int(b['ymax'])), (0, 255, 0), 2)
        
        # Calculate aiming coordinates
        cx, cy = int((b['xmin'] + b['xmax']) / 2), int((b['ymin'] + b['ymax']) / 2)
        
        # Crosshair and status text
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(frame, f"LOCKED: {res['label']} ({round(res['score'], 2)})", 
                    (int(b['xmin']), int(b['ymin']-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Custom Trained Turret Vision', frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
        
"""
To improve this for your project:

Skip Frames: Only run the AI on every 3rd frame and use the last known coordinates for the other 2 frames. (Implemented above)

Resize: Shrink the image before sending it to the model (pil_img.thumbnail((320, 320))) and then scale the coordinates back up. (Implemented above)
"""