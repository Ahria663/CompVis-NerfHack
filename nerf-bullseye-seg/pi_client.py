"""
Nerf Turret - Pi Zero 2 W Client (libcamera)
Motion detection + API calls to server

Uses Picamera2 (libcamera) instead of cv2.VideoCapture, because on
modern Raspberry Pi OS the CSI camera usually isn't exposed as /dev/video0.
"""

import time
import requests
import numpy as np
import cv2
from collections import deque

from picamera2 import Picamera2

# ============================================================
# SETTINGS YOU NEED TO CHANGE
# ============================================================

API_URL = "http://YOUR_SERVER_IP:8000/detect"  # <-- change this

# Motion detection settings
MOTION_THRESHOLD = 3000      # Lower = more sensitive
COOLDOWN_SECONDS = 0.5       # Minimum time between API calls

# Frame settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80            # Lower = smaller file, faster upload

# Motion pre-processing (helps reduce noise)
BLUR_KERNEL = (7, 7)         # (5,5) or (7,7) good; keep odd numbers
USE_GRAYSCALE_FOR_MOTION = True

# ============================================================


class PiTurretClient:
    def __init__(self):
        # Background subtractor works best on a single channel (gray) image
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

        self.picam2 = None

    # ---------------------------
    # Camera setup (Picamera2)
    # ---------------------------
    def setup_camera(self):
        self.picam2 = Picamera2()

        # RGB888 gives a nice numpy array format
        config = self.picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)  # let exposure settle a bit

    def read_frame(self):
        """
        Returns a BGR frame compatible with OpenCV.
        Picamera2 returns RGB by default for RGB888.
        """
        rgb = self.picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    # ---------------------------
    # Motion detection
    # ---------------------------
    def detect_motion(self, frame_bgr):
        """
        Returns True if significant motion detected.
        We feed grayscale+blur to reduce flicker/noise.
        """
        if USE_GRAYSCALE_FOR_MOTION:
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = frame_bgr

        img = cv2.GaussianBlur(img, BLUR_KERNEL, 0)

        fg_mask = self.bg_subtractor.apply(img)
        motion_pixels = cv2.countNonZero(fg_mask)
        return motion_pixels > MOTION_THRESHOLD

    # ---------------------------
    # API call
    # ---------------------------
    def call_api(self, frame_bgr):
        """
        Send frame to server, get target info.
        """
        ok, img_encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            print("JPEG encode failed")
            return None

        try:
            response = requests.post(
                API_URL,
                files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                timeout=5
            )
            self.api_calls += 1

            # Helpful debug if server errors:
            # print("Status:", response.status_code, "Body:", response.text[:200])

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API error: {e}")
            return None
        except ValueError:
            # JSON decode error
            print("API error: server did not return valid JSON")
            return None

    # ---------------------------
    # Smoothing helpers
    # ---------------------------
    def get_smooth_center(self):
        """Average recent centers for smooth aiming."""
        if len(self.center_history) == 0:
            return None
        x = int(np.mean([c[0] for c in self.center_history]))
        y = int(np.mean([c[1] for c in self.center_history]))
        return (x, y)

    # ---------------------------
    # Detection handling
    # ---------------------------
    def process_detection(self, result):
        """
        Handle detection result - ADD YOUR SERVO CODE HERE.
        Expected result shape (example):
          {
            "found": true,
            "center": [x,y],
            "frame_center": [cx,cy],
            "human_detected": false
          }
        """
        if not isinstance(result, dict):
            print("Bad result (not a dict)")
            return

        if result.get("human_detected"):
            print("!!! HUMAN DETECTED - SAFETY LOCK !!!")
            # TODO: Disable servos / safe mode
            return

        if result.get("found"):
            self.detections += 1
            center = result.get("center")
            if not center or len(center) != 2:
                print("Result said found, but center missing/invalid")
                return

            center = (int(center[0]), int(center[1]))
            self.center_history.append(center)

            smooth = self.get_smooth_center()
            frame_center = result.get("frame_center", [FRAME_WIDTH // 2, FRAME_HEIGHT // 2])
            frame_center = (int(frame_center[0]), int(frame_center[1]))

            offset_x = smooth[0] - frame_center[0]
            offset_y = smooth[1] - frame_center[1]

            print(f"TARGET: {smooth}, offset: ({offset_x}, {offset_y})")

            # ================================================
            # TODO: ADD YOUR SERVO CONTROL CODE HERE
            # Example:
            # if offset_x > 20:
            #     pan_servo.move_right()
            # elif offset_x < -20:
            #     pan_servo.move_left()
            #
            # if offset_y > 20:
            #     tilt_servo.move_down()
            # elif offset_y < -20:
            #     tilt_servo.move_up()
            # ================================================
        else:
            print("No target")

    # ---------------------------
    # Main loop
    # ---------------------------
    def run(self):
        print("=" * 50)
        print("NERF TURRET - PI ZERO 2 W CLIENT (Picamera2)")
        print("=" * 50)
        print(f"API: {API_URL}")
        print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print("=" * 50)

        try:
            self.setup_camera()
        except Exception as e:
            print(f"ERROR: Cannot start camera (Picamera2): {e}")
            return

        print("✓ Camera ready (libcamera/Picamera2)")
        print("✓ Monitoring for motion...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                frame = self.read_frame()
                self.frame_count += 1

                if self.detect_motion(frame):
                    now = time.time()

                    if now - self.last_api_call > COOLDOWN_SECONDS:
                        result = self.call_api(frame)
                        if result is not None:
                            self.process_detection(result)

                        self.last_api_call = now

                if self.frame_count % 100 == 0:
                    print(f"Frames: {self.frame_count}, API calls: {self.api_calls}, Detections: {self.detections}")

                time.sleep(0.05)  # ~20 FPS motion check

        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            try:
                if self.picam2 is not None:
                    self.picam2.stop()
            except Exception:
                pass

            print(f"\nStats: {self.frame_count} frames, {self.api_calls} API calls, {self.detections} detections")


if __name__ == "__main__":
    client = PiTurretClient()
    client.run()
