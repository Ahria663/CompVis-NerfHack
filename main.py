import cv2
import numpy as np

# Define the Green HSV range
# Hue: 35-85 (covers most shades of green)
# Saturation: 100-255 (excludes grey/white)
# Value: 40-255 (excludes pure black)
LOWER_GREEN = np.array([35, 100, 40])
UPPER_GREEN = np.array([85, 255, 255])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Pre-process: Blur to reduce noise and convert to HSV
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. Create the Mask
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    
    # 3. Clean up the mask (remove small noise dots)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 4. Find the outlines (contours) of green objects
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Only detect items larger than 500 pixels (prevents tiny flicker)
        if area > 500:
            # Get the square coordinates
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw the box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add the label
            cv2.putText(frame, "nerf target", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Center Crosshair
            cx, cy = x + w//2, y + h//2
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

    # Show the results
    cv2.imshow("Green Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()