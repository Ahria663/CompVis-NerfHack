import cv2
import torch
import numpy as npq
from transformers import pipeline
from PIL import Image  

# --- COLOR VERIFICATION FUNCTION ---
def is_it_red(frame, box):
    """
    TWEAK HERE: To change target color, adjust the 'lower' and 'upper' HSV arrays.
    This function acts as a 'second opinion' for the AI to ensure the target is red.
    """
    # Convert AI coordinates to integers and ensure they stay within the camera frame
    xmin, ymin = max(0, int(box['xmin'])), max(0, int(box['ymin']))
    xmax, ymax = min(frame.shape[1], int(box['xmax'])), min(frame.shape[0], int(box['ymax']))
    
    # ROI (Region of Interest): Crop the image to only show what the AI detected
    roi = frame[ymin:ymax, xmin:xmax]
    
    # Guard against empty detections
    if roi.size == 0: return False

    # Convert crop from BGR (standard) to HSV (better for color isolation)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # RED logic: Red is unique because it exists at both the start and end of the HSV spectrum.
    # lower_red1/upper_red1: Darker reds (0-10 degrees)
    # lower_red2/upper_red2: Brighter/Pinkish reds (170-180 degrees)
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    
    # Create a binary mask: pixels matching red become white (255), everything else black (0)
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # TWEAK HERE: Increase '0.15' (15%) if the turret is firing at things that are only slightly red.
    return (np.sum(mask > 0) / mask.size) > 0.15 

# --- AI INITIALIZATION ---
# Select GPU (cuda) if you have one, otherwise use CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the DETR (Detection Transformer). This is a modern, non-RNN architecture.
detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=device)

# --- CONFIGURATION (TWEAK FREQUENTLY) ---
# TWEAK HERE: Add labels the AI might call your target (e.g., 'frisbee' for a round target).
MY_TARGETS = ['sports ball', 'cup', 'bottle', 'apple', 'orange']

# TWEAK HERE: Lower this (e.g., 0.3) if the AI is missing targets. Raise it (e.g., 0.8) if it's too twitchy.
MIN_CONFIDENCE = 0.5

# TWEAK HERE: Higher number = faster video but slower AI updates. 4-6 is usually a sweet spot for laptops.
SKIP_FRAMES = 4 

cap = cv2.VideoCapture(0)
frame_count = 0
last_results = []

print("System Active. Searching for RED targets...")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape

    # Performance optimization: The 'Brain' only thinks every X frames to keep the 'Eyes' smooth.
    if frame_count % SKIP_FRAMES == 0:
        # Prep image for AI: DETR works better with RGB (PIL) than BGR (OpenCV)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Shrinking image makes the Transformer math run significantly faster.
        pil_img.thumbnail((400, 400))
        
        # Run inference (The actual AI detection)
        raw_results = detector(pil_img, threshold=MIN_CONFIDENCE)

        valid_targets = []
        for res in raw_results:
            # FILTER 1: Label check (Stops it from shooting people)
            if res['label'] in MY_TARGETS:
                
                # RE-SCALING: Because we resized to 400x400 for the AI, we must scale coords back 
                # to the original camera resolution (e.g., 640x480) for accurate aiming.
                box = res['box']
                scale_x, scale_y = w / 400, h / 400
                scaled_box = {
                    'xmin': box['xmin'] * scale_x, 'ymin': box['ymin'] * scale_y,
                    'xmax': box['xmax'] * scale_x, 'ymax': box['ymax'] * scale_y
                }
                
                # FILTER 2: Color check (Stops it from shooting non-red objects in your list)
                if is_it_red(frame, scaled_box):
                    res['scaled_box'] = scaled_box 
                    valid_targets.append(res)

        # SELECTION LOGIC (The 'Suggestible' part):
        # We only care about ONE target. Here we pick the one the AI is most confident in.
        # TWEAK HERE: Change 'max' to find the target closest to center if you want a different priority.
        last_results = [max(valid_targets, key=lambda x: x['score'])] if valid_targets else []

    # --- UI DRAWING (Visual Feedback) ---
    for res in last_results:
        b = res['scaled_box']
        # Draw Red Bounding Box
        cv2.rectangle(frame, (int(b['xmin']), int(b['ymin'])), (int(b['xmax']), int(b['ymax'])), (0, 0, 255), 3)
        
        # Calculate the Target Center (This is the coordinate you send to the Raspberry Pi)
        cx, cy = int((b['xmin'] + b['xmax']) / 2), int((b['ymin'] + b['ymax']) / 2)
        
        # Draw Target Crosshair
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"LOCKED: {res['label']}", (int(b['xmin']), int(b['ymin']-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Turret Vision', frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
        
"""
To improve this for your project:

Skip Frames: Only run the AI on every 3rd frame and use the last known coordinates for the other 2 frames. (Implemented above)

Resize: Shrink the image before sending it to the model (pil_img.thumbnail((320, 320))) and then scale the coordinates back up. (Implemented above)
"""