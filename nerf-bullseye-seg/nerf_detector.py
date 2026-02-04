#!/usr/bin/env python3
"""
Nerf Hackathon Target Detector
Detects and aims at water bottles using YOLOv8
Outputs center coordinates for servo control
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import time
import sys
import threading
import os
from datetime import datetime
import argparse


class TargetDetector:
    """Competition-ready blue object detector using YOLOv8 + color filter"""
    
    def __init__(self, smooth_frames: int = 5, reconnect_attempts: int = 5, reconnect_delay: float = 2.0,
                 conf: float = 0.3, min_area: int = 200, h_min: int = 100, h_max: int = 130):
        """Initialize detector with YOLOv8 model and green color filter

        reconnect_attempts: number of times to try reopening network stream
        reconnect_delay: base delay (seconds) between reconnect attempts (exponential backoff)
        """
        self.frame_count = 0
        self.detected_count = 0
        self.smooth_frames = smooth_frames
        self.center_history = deque(maxlen=smooth_frames)
        self.last_center = None
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.conf = conf
        self.min_area = min_area
        self.h_min = h_min
        self.h_max = h_max
        self.camera_source = 0
        self.cap = None
        self._cap_lock = threading.Lock()
        self._reconnect_thread = None
        self._reconnect_running = False
        self.last_frame = None
        
        # Load YOLOv8 model
        print("Loading YOLOv8 model for blue object detection...")
        self.model = YOLO('yolov8n.pt')  # Nano model for speed
        print("✓ YOLOv8 model loaded")
    
    def detect_target(self, frame):
        target = {
            'found': False,
            'center': None,
            'area': 0,
            'bbox': None,
            'confidence': 0,
            'human_in_frame': False  # Track if a person is present
        }
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # --- THE PRECISION GREEN FILTER ---
        # Hue: 35 to 80 (Strictly cuts off before Teal/Blue begins at 85+)
        # Saturation: 60 to 255 (Cuts off Blacks, Grays, and 'Muddy' dark colors)
        # Value: 40 to 255 (Cuts off deep shadows that confuse the sensor)
        lower_green = np.array([35, 60, 40]) 
        upper_green = np.array([80, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Optional: Clean up noise (small specks)
        kernel = np.ones((5,5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        results = self.model(frame, verbose=False, conf=self.conf)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            valid_detections = []
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # STRICT HUMAN AVOIDANCE
                if cls == 0:
                    target['human_in_frame'] = True
                    continue 
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # Check green content in the YOLO box
                roi = green_mask[y1:y2, x1:x2]
                if roi.size > 0:
                    green_ratio = np.sum(roi > 0) / roi.size
                    
                    # If it's a known object (YOLO) AND has green pixels, lock on
                    if green_ratio > 0.05: # 5% minimum green content
                        area = (x2 - x1) * (y2 - y1)
                        if area > self.min_area:
                            valid_detections.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'area': area, 'conf': conf
                            })
            
            if valid_detections:
                best = max(valid_detections, key=lambda x: x['conf'])
                cx, cy = int((best['x1'] + best['x2']) / 2), int((best['y1'] + best['y2']) / 2)
                
                target.update({
                    'found': True,
                    'center': (cx, cy),
                    'confidence': best['conf'],
                    'area': best['area'],
                    'bbox': (best['x1'], best['y1'], best['x2'] - best['x1'], best['y2'] - best['y1'])
                })
                self.center_history.append((cx, cy))
        
        return target
    
    def get_smooth_center(self):
        """Get smoothed center from history"""
        if len(self.center_history) == 0:
            return None
        
        # Average of recent frames
        x = int(np.mean([c[0] for c in self.center_history]))
        y = int(np.mean([c[1] for c in self.center_history]))
        return (x, y)
    
    def process_frame(self, frame):
        """Process single frame"""
        self.frame_count += 1
        
        target = self.detect_target(frame)
        
        if target['found']:
            self.detected_count += 1
            self.last_center = target['center']
        
        # Get smoothed center
        smooth_center = self.get_smooth_center()
        
        return self._annotate(frame, target, smooth_center)
    
    def _annotate(self, frame, target, smooth_center):
        """Draw aiming reticle and info"""
        display = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center crosshairs
        cv2.line(display, (w//2 - 50, h//2), (w//2 + 50, h//2), (0, 255, 255), 2)
        cv2.line(display, (w//2, h//2 - 50), (w//2, h//2 + 50), (0, 255, 255), 2)
        cv2.circle(display, (w//2, h//2), 20, (0, 255, 255), 1)
        
        if target['found']:
            # Draw target box
            x, y, bw, bh = target['bbox']
            cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
            # Draw target center
            cx, cy = target['center']
            cv2.circle(display, (cx, cy), 60, (0, 255, 0), 3)
            cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
            
            # Draw smooth center if available
            if smooth_center:
                cv2.circle(display, smooth_center, 80, (255, 0, 0), 2)
            
            # Status
            cv2.putText(display, "(target detected)", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)
            cv2.putText(display, f"Position: {target['center']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Area: {target['area']}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if smooth_center:
                cv2.putText(display, f"Smoothed: {smooth_center}", (10, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(display, "NO TARGET", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)
        
        if target.get('human_in_frame'):
            cv2.putText(display, "!!! HUMAN DETECTED - SAFETY LOCK !!!", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Stats
        if self.frame_count > 0:
            rate = 100 * self.detected_count / self.frame_count
            cv2.putText(display, f"Detection: {rate:.1f}%", 
                       (display.shape[1] - 300, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        cv2.putText(display, "Press 'q' to quit, 's' to save", 
                   (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return display
    
    def run_live(self, camera_source=0):
        """Run live targeting.

        `camera_source` may be an int (camera index) or a string URL (ESP32-CAM stream).
        Example URL formats: `http://<ESP_IP>:81/` or `http://<ESP_IP>/stream` depending on firmware.
        """
        print("\n" + "="*70)
        print("NERF HACKATHON TARGET DETECTOR")
        print("="*70)
        print("\nOpening camera for target acquisition...\n")
        # If a numeric string is passed, convert to int
        source = camera_source
        try:
            # allow numeric strings
            if isinstance(camera_source, str) and camera_source.isdigit():
                source = int(camera_source)
        except Exception:
            source = camera_source

        # Try to open capture with retries for network streams
        cap = None
        for attempt in range(self.reconnect_attempts):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                with self._cap_lock:
                    self.cap = cap
                self.camera_source = source
                break
            print(f"Warning: unable to open camera source {source}, retrying ({attempt+1}/{self.reconnect_attempts})...")
            time.sleep(self.reconnect_delay * (attempt + 1))
        
        
        if self.cap is None or (not self.cap.isOpened()):
            print("ERROR: Cannot open camera! Ensure the ESP32-CAM stream URL or camera index is correct and reachable.")
            return

        # Start background reconnect monitor for network sources
        self._start_reconnect_monitor()
        
        print("✓ Camera ready")
        print("✓ Point at target to lock on")
        print("✓ Blue circle = target center")
        print("✓ Yellow crosshair = camera center")
        print("✓ Cyan circle = smoothed position (for aiming)")
        print("\nPress 'q' to quit\n")
        print("="*70 + "\n")
        
        while True:
            with self._cap_lock:
                local_cap = self.cap

            if local_cap is None:
                print("ERROR: Capture object missing; exiting live loop.")
                break

            ret, frame = local_cap.read()
            if not ret or frame is None:
                # Save last debug frame if available and attempt reconnects happen in background
                print("Warning: frame read failed — waiting for reconnect monitor to recover...")
                # wait briefly to give reconnect thread time to act
                time.sleep(0.5)
                continue
            # store last good frame for debug saving on reconnect failure
            with self._cap_lock:
                try:
                    self.last_frame = frame.copy()
                except Exception:
                    self.last_frame = None
            
            display = self.process_frame(frame)
            cv2.imshow("Target Detector", display)
            
            # Print status every 60 frames
            if self.frame_count % 60 == 0:
                rate = 100 * self.detected_count / self.frame_count if self.frame_count > 0 else 0.0
                print(f"Frame {self.frame_count}: {rate:.1f}% lock rate")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"target_{self.frame_count}.jpg"
                cv2.imwrite(filename, display)
                print(f"Saved: {filename}")
        
        # Stop reconnect monitor and release capture
        self._stop_reconnect_monitor()
        with self._cap_lock:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        
        print(f"\n{'='*70}")
        print("HACKATHON STATS")
        print(f"{'='*70}")
        print(f"Total frames: {self.frame_count}")
        print(f"Targets detected: {self.detected_count}")
        if self.frame_count > 0:
            print(f"Lock rate: {100*self.detected_count/self.frame_count:.1f}%")
        print(f"{'='*70}\n")

    # --- Reconnect monitor ---
    def _start_reconnect_monitor(self):
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return
        self._reconnect_running = True
        self._reconnect_thread = threading.Thread(target=self._reconnect_worker, daemon=True)
        self._reconnect_thread.start()

    def _stop_reconnect_monitor(self):
        self._reconnect_running = False
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=2.0)
            self._reconnect_thread = None

    def _reconnect_worker(self):
        """Background worker that attempts to reopen the camera if connection lost."""
        backoff_base = self.reconnect_delay
        while self._reconnect_running:
            with self._cap_lock:
                cap = self.cap
                source = self.camera_source

            if cap is None or not cap.isOpened():
                # Attempt reconnects
                for attempt in range(self.reconnect_attempts):
                    if not self._reconnect_running:
                        break
                    print(f"[reconnect] attempt {attempt+1}/{self.reconnect_attempts} to open {source}")
                    new_cap = cv2.VideoCapture(source)
                    if new_cap.isOpened():
                        with self._cap_lock:
                            # release old cap if exists
                            try:
                                if self.cap is not None:
                                    self.cap.release()
                            except Exception:
                                pass
                            self.cap = new_cap
                        print("[reconnect] camera reopened")
                        break
                    else:
                        try:
                            new_cap.release()
                        except Exception:
                            pass
                    time.sleep(backoff_base * (attempt + 1))

                # If still not opened after attempts, save a debug marker
                with self._cap_lock:
                    still_closed = (self.cap is None) or (not self.cap.isOpened())
                if still_closed:
                    debug_dir = 'debug_frames'
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    marker = os.path.join(debug_dir, f'reconnect_failed_{timestamp}.txt')
                    with open(marker, 'w') as f:
                        f.write(f"Reconnect failed at {timestamp} for source {source}\n")
                    print(f"[reconnect] saved marker {marker}")
                    # Save last good frame if we have one
                    with self._cap_lock:
                        lf = self.last_frame.copy() if self.last_frame is not None else None
                    if lf is not None:
                        imgpath = os.path.join(debug_dir, f'last_frame_{timestamp}.jpg')
                        try:
                            cv2.imwrite(imgpath, lf)
                            print(f"[reconnect] saved last frame {imgpath}")
                        except Exception:
                            print("[reconnect] failed to save last frame image")
            time.sleep(1.0)


def _parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 blue object detector (ESP32-CAM friendly)')
    parser.add_argument('camera_source', nargs='?', default='0',
                        help='Camera index or stream URL (e.g. http://<ESP_IP>:81/)')
    parser.add_argument('--smooth', type=int, default=5, help='Smoothing frames')
    parser.add_argument('--reconnect-attempts', type=int, default=5, help='Reconnect attempts')
    parser.add_argument('--reconnect-delay', type=float, default=2.0, help='Base reconnect delay (seconds)')
    parser.add_argument('--conf', type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--min-area', type=int, default=200, help='Minimum detection area in pixels')
    parser.add_argument('--h-min', type=int, default=100, help='HSV Hue min (0-180)')
    parser.add_argument('--h-max', type=int, default=130, help='HSV Hue max (0-180)')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cam_src = args.camera_source
    # convert numeric strings to int
    if isinstance(cam_src, str) and cam_src.isdigit():
        cam_src = int(cam_src)

    detector = TargetDetector(smooth_frames=args.smooth,
                              reconnect_attempts=args.reconnect_attempts,
                              reconnect_delay=args.reconnect_delay,
                              conf=args.conf,
                              min_area=args.min_area,
                              h_min=args.h_min,
                              h_max=args.h_max)
    detector.run_live(camera_source=cam_src)
    
import psutil

import shutil

def check_system_status():
    # 1. Power/Battery Usage
    battery = psutil.sensors_battery()
    if battery:
        plugged = "Plugged In" if battery.power_plugged else "Discharged"
        print(f"--- Power Status ---")
        print(f"Battery Level: {battery.percent}%")
        print(f"Status:        {plugged}")
        print(f"Time Left:     {battery.secsleft // 60} minutes (approx)")
    else:
        print("--- Power Status ---\nBattery info not available (Desktop PC?)")

    print("\n--- Capacity Usage ---")
    
    # 2. CPU Usage
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"CPU Usage:     {cpu_usage}%")

    # 3. RAM Usage
    memory = psutil.virtual_memory()
    print(f"RAM Usage:     {memory.percent}% ({memory.used // (1024**2)}MB / {memory.total // (1024**2)}MB)")

    # 4. Disk Usage
    total, used, free = shutil.disk_usage("/")
    print(f"Disk Usage:    {(used/total)*100:.1f}% used")
    print(f"Disk Free:     {free // (1024**3)} GB remaining")

if __name__ == "__main__":
    check_system_status()
