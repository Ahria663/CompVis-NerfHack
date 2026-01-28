import cv2
import os

# Path to your raw photos folder
INPUT_FOLDER = "./PHOTOS"
OUTPUT_FOLDER = "./PREPPED_PHOTOS"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"Starting prep for photos in {INPUT_FOLDER}...")

count = 0
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Resize to a square 640x640 (standard for DETR/YOLO)
            # This makes training faster and more accurate
            resized = cv2.resize(img, (640, 640))
            
            # Save with a clean name: target_001.jpg, target_002.jpg, etc.
            new_name = f"target_{str(count).zfill(3)}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, new_name), resized)
            count += 1

print(f"Finished! {count} photos are ready in {OUTPUT_FOLDER}")