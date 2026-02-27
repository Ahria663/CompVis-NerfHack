"""
Nerf Turret - Pi Zero 2 W Client (libcamera)
Motion detection + API calls to server + LIVE PREVIEW OVERLAY

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
# PREVIEW / OVERLAY SETTINGS
# ============================================================

ENABLE_PREVIEW = True            # Set False if running totally headless
WINDOW_NAME = "Turret Cam"
SHOW_MOTION_MASK = False         # Show the foreground mask in a second window
DRAW_CROSSHAIR = True
BOX_HOLD_SECONDS = 0.35          # Keep last bbox visible for this long after last API response
FONT_SCALE = 0.6
TEXT_THICKNESS = 2

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

        self.picam2 = None

        # For overlay / GUI
        self.preview_enabled = ENABLE_PREVIEW
        self.last_result = None
        self.last_result_time = 0.0
        self.last_motion_pixels = 0

        # FPS tracking
        self._fps_last_t = time.time()
        self._fps_counter = 0
        self.fps = 0.0

    # ---------------------------
    # Camera setup (Picamera2)
    # ---------------------------
    def setup_camera(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)

    def read_frame(self):
        rgb = self.picam2.capture_array()
        bgr = rgb[:, :,1]
        return bgr

    # ---------------------------
    # Motion detection
    # ---------------------------
    def detect_motion(self, frame_bgr):
        if USE_GRAYSCALE_FOR_MOTION:
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = frame_bgr

        img = cv2.GaussianBlur(img, BLUR_KERNEL, 0)
        fg_mask = self.bg_subtractor.apply(img)

        motion_pixels = cv2.countNonZero(fg_mask)
        self.last_motion_pixels = motion_pixels

        if SHOW_MOTION_MASK and self.preview_enabled:
            cv2.imshow("Motion Mask", fg_mask)

        return motion_pixels > MOTION_THRESHOLD

    # ---------------------------
    # API call
    # ---------------------------
    def call_api(self, frame_bgr):
        ok, img_encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            print("JPEG encode failed")
            return None

        try:
            t0 = time.time()
            response = requests.post(
                API_URL,
                files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                timeout=5
            )
            self.api_calls += 1
            response.raise_for_status()
            data = response.json()

            # Attach simple timing info (optional)
            data["_client_rtt_ms"] = int((time.time() - t0) * 1000)
            return data

        except requests.exceptions.RequestException as e:
            print(f"API error: {e}")
            return None
        except ValueError:
            print("API error: server did not return valid JSON")
            return None

    # ---------------------------
    # Smoothing helpers
    # ---------------------------
    def get_smooth_center(self):
        if len(self.center_history) == 0:
            return None
        x = int(np.mean([c[0] for c in self.center_history]))
        y = int(np.mean([c[1] for c in self.center_history]))
        return (x, y)

    # ---------------------------
    # Detection handling
    # ---------------------------
    def process_detection(self, result):
        if not isinstance(result, dict):
            print("Bad result (not a dict)")
            return

        # Save for overlay
        self.last_result = result
        self.last_result_time = time.time()

        if result.get("human_detected"):
            print("!!! HUMAN DETECTED - SAFETY LOCK !!!")
            return

        if result.get("found"):
            self.detections += 1

            center = result.get("center")
            if center and len(center) == 2:
                center = (int(center[0]), int(center[1]))
                self.center_history.append(center)

            smooth = self.get_smooth_center()
            frame_center = result.get("frame_center", [FRAME_WIDTH // 2, FRAME_HEIGHT // 2])
            frame_center = (int(frame_center[0]), int(frame_center[1]))

            if smooth:
                offset_x = smooth[0] - frame_center[0]
                offset_y = smooth[1] - frame_center[1]
                print(f"TARGET: {smooth}, offset: ({offset_x}, {offset_y})")
            else:
                print("Target found, but smoothing not ready yet")
        else:
            print("No target")

    # ---------------------------
    # Overlay drawing
    # ---------------------------
    def draw_overlay(self, frame):
        # Update FPS
        self._fps_counter += 1
        now = time.time()
        if now - self._fps_last_t >= 1.0:
            self.fps = self._fps_counter / (now - self._fps_last_t)
            self._fps_counter = 0
            self._fps_last_t = now

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        if DRAW_CROSSHAIR:
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)

        # Decide whether to draw last bbox (hold for BOX_HOLD_SECONDS)
        draw_result = False
        if self.last_result and (now - self.last_result_time) <= BOX_HOLD_SECONDS:
            draw_result = True

        status_line = f"FPS {self.fps:.1f} | motion {self.last_motion_pixels} | API {self.api_calls} | det {self.detections}"

        # Draw detection box / center
        if draw_result:
            r = self.last_result

            if r.get("human_detected"):
                cv2.putText(frame, "HUMAN DETECTED - SAFETY LOCK",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif r.get("found"):
                bbox = r.get("bbox")  # server uses [x, y, w, h]
                conf = r.get("confidence", 0)
                rtt = r.get("_client_rtt_ms", None)

                if bbox and len(bbox) == 4:
                    x, y, bw, bh = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

                    label = f"target conf={conf:.2f}"
                    if rtt is not None:
                        label += f" rtt={rtt}ms"

                    cv2.putText(frame, label,
                                (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), TEXT_THICKNESS)

                center = r.get("center")
                if center and len(center) == 2:
                    tx, ty = int(center[0]), int(center[1])
                    cv2.circle(frame, (tx, ty), 5, (255, 0, 0), -1)

        # Bottom status bar
        cv2.putText(frame, status_line,
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), TEXT_THICKNESS)

        return frame

    # ---------------------------
    # Main loop
    # ---------------------------
    def run(self):
        print("=" * 60)
        print("NERF TURRET - PI ZERO 2 W CLIENT (Picamera2 + Preview)")
        print("=" * 60)
        print(f"API: {API_URL}")
        print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print(f"Preview: {self.preview_enabled} (press 'q' to quit preview)")
        print("=" * 60)

        try:
            self.setup_camera()
        except Exception as e:
            print(f"ERROR: Cannot start camera (Picamera2): {e}")
            return

        print("✓ Camera ready (libcamera/Picamera2)")
        print("✓ Monitoring for motion...")
        print("Press Ctrl+C to stop\n")

        # If preview is enabled but no display exists, OpenCV may error.
        # We'll try once and disable preview if it fails.
        if self.preview_enabled:
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            except Exception as e:
                print(f"Preview disabled (no GUI/display available): {e}")
                self.preview_enabled = False

        try:
            while True:
                frame = self.read_frame()
                self.frame_count += 1

                # Motion detection + API call
                if self.detect_motion(frame):
                    now = time.time()
                    if now - self.last_api_call > COOLDOWN_SECONDS:
                        result = self.call_api(frame)
                        if result is not None:
                            self.process_detection(result)
                        self.last_api_call = now

                # Preview overlay
                if self.preview_enabled:
                    out = self.draw_overlay(frame)
                    cv2.imshow(WINDOW_NAME, out)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                # Occasional status in terminal
                if self.frame_count % 200 == 0:
                    print(f"Frames: {self.frame_count}, API calls: {self.api_calls}, Detections: {self.detections}")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            try:
                if self.picam2 is not None:
                    self.picam2.stop()
            except Exception:
                pass

            if self.preview_enabled:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            print(f"\nStats: {self.frame_count} frames, {self.api_calls} API calls, {self.detections} detections")


if __name__ == "__main__":
    client = PiTurretClient()
    client.run()
