"""
Nerf Turret - Pi Zero Client
Motion detection + API calls to server
"""

import cv2
import requests
import time
import numpy as np
from collections import deque

# ============================================================
# SETTINGS YOU NEED TO CHANGE
# ============================================================

# Your server URL - change this!
# Examples:
#   Local network: "http://192.168.1.100:8000/detect"
#   Hugging Face:  "https://your-space.hf.space/detect"
API_URL = "http://YOUR_SERVER_IP:8000/detect"

# Camera source
# Use 0 for USB camera, or ESP32-CAM URL like "http://192.168.1.50:81/stream"
CAMERA_SOURCE = 0

# Motion detection settings
MOTION_THRESHOLD = 3000      # Lower = more sensitive
COOLDOWN_SECONDS = 0.5       # Minimum time between API calls

# Frame settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80            # Lower = smaller file, faster upload

# ============================================================


class PiTurretClient:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )
        self.last_api_call = 0
        self.center_history = deque(maxlen=5)
        self.frame_count = 0
        self.api_calls = 0
        self.detections = 0
    
    def detect_motion(self, frame):
        """Returns True if significant motion detected"""
        fg_mask = self.bg_subtractor.apply(frame)
        motion_pixels = cv2.countNonZero(fg_mask)
        return motion_pixels > MOTION_THRESHOLD
    
    def call_api(self, frame):
        """Send frame to server, get target info"""
        _, img_encoded = cv2.imencode(
            '.jpg', frame, 
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        
        try:
            response = requests.post(
                API_URL,
                files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                timeout=5
            )
            self.api_calls += 1
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API error: {e}")
            return None
    
    def get_smooth_center(self):
        """Average recent centers for smooth aiming"""
        if len(self.center_history) == 0:
            return None
        x = int(np.mean([c[0] for c in self.center_history]))
        y = int(np.mean([c[1] for c in self.center_history]))
        return (x, y)
    
    def process_detection(self, result):
        """Handle detection result - ADD YOUR SERVO CODE HERE"""
        if result.get("human_detected"):
            print("!!! HUMAN DETECTED - SAFETY LOCK !!!")
            # TODO: Disable servos / safe mode
            return
        
        if result.get("found"):
            self.detections += 1
            center = result["center"]
            self.center_history.append(center)
            
            smooth = self.get_smooth_center()
            frame_center = result.get("frame_center", [320, 240])
            
            # Calculate offset from center
            offset_x = center[0] - frame_center[0]
            offset_y = center[1] - frame_center[1]
            
            print(f"TARGET: {center}, offset: ({offset_x}, {offset_y})")
            
            # ================================================
            # TODO: ADD YOUR SERVO CONTROL CODE HERE
            # Example:
            #   if offset_x > 20:
            #       pan_servo.move_right()
            #   elif offset_x < -20:
            #       pan_servo.move_left()
            # ================================================
        else:
            print("No target")
    
    def run(self):
        """Main loop"""
        print("="*50)
        print("NERF TURRET - PI CLIENT")
        print("="*50)
        print(f"API: {API_URL}")
        print(f"Camera: {CAMERA_SOURCE}")
        print("="*50)
        
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return
        
        print("✓ Camera ready")
        print("✓ Monitoring for motion...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame read failed")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Check for motion
                if self.detect_motion(frame):
                    now = time.time()
                    
                    # Respect cooldown
                    if now - self.last_api_call > COOLDOWN_SECONDS:
                        result = self.call_api(frame)
                        
                        if result:
                            self.process_detection(result)
                        
                        self.last_api_call = now
                
                # Status every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"Frames: {self.frame_count}, API calls: {self.api_calls}, Detections: {self.detections}")
                
                time.sleep(0.05)  # ~20 FPS motion check
                
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            cap.release()
            print(f"\nStats: {self.frame_count} frames, {self.api_calls} API calls, {self.detections} detections")


if __name__ == "__main__":
    client = PiTurretClient()
    client.run()