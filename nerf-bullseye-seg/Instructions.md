# Nerf Hackathon — nerf_detector.py

This README explains what `nerf_detector.py` does, how to run it, and how to tune its parameters. It's written as a short instructional guide for students building and testing the detector.

## Overview
- Purpose: runs a fast object detector (YOLOv8-nano) combined with an HSV color filter to locate a single target in a camera stream (USB camera or ESP32-CAM). The program smooths target position, displays a live view with aiming reticles, and enforces a human-safety lock.
- Output: annotated video window and printed statistics (frames, lock rate); the code provides center coordinates that can be consumed by servo code.

## Quick Start (example)
1. Activate your Python environment (where `ultralytics`, `opencv-python`, and `numpy` are installed):

```bash
source .venv/bin/activate
python nerf_detector.py 0 --conf 0.3 --min-area 200 --smooth 5 --reconnect-attempts 5 --reconnect-delay 2.0
```

Or use an ESP32-CAM HTTP stream (replace `<ESP_IP>`):

```bash
python nerf_detector.py "http://<ESP_IP>:81/" --conf 0.35 --min-area 300
```

Notes:
- `camera_source` accepts a numeric camera index (`0`, `1`, ...) or a URL string.
- Press `q` in the display window to quit, `s` to save the current display image.

## CLI Options
- `camera_source` (positional): Camera index or stream URL (default `0`).
- `--smooth INT` (default 5): Number of recent centers averaged for smoothing.
- `--reconnect-attempts INT` (default 5): How many times the background reconnect worker tries before writing a failure marker.
- `--reconnect-delay FLOAT` (default 2.0): Base delay (seconds) used for exponential backoff between reconnect attempts.
- `--conf FLOAT` (default 0.3): YOLOv8 confidence threshold.
- `--min-area INT` (default 200): Minimum bounding-box area (pixels) for a detection to be considered.
- `--h-min INT` and `--h-max INT` (defaults present): Intended for HSV hue bounds (see note below).

IMPORTANT: In the current `nerf_detector.py` implementation the HSV color filter for the target is coded using fixed `lower_green` / `upper_green` values. The `--h-min`/`--h-max` CLI flags are defined but not currently wired to the mask in the `detect_target()` routine. See "Tuning" for how to change the HSV range.

## How It Works (step-by-step)
1. Camera open: tries to open the given `camera_source` with startup retries.
2. YOLOv8 model: loads `yolov8n.pt` (nano) for fast CPU inference.
3. Per-frame processing:
   - Convert the frame to HSV and apply a color mask (current code uses a green filter).
   - Run YOLO on the full frame to get object boxes + classes + confidences.
   - Human avoidance: if YOLO reports class `0` (person) in the frame, the code sets a safety lock for that frame and will not target other objects.
   - For each YOLO box (non-human): compute the overlap ratio of the box with the color mask (how much of the box is target-colored).
   - Keep detections that have >5% color ratio and bounding-box area > `--min-area`.
   - Choose the detection with highest YOLO confidence, compute its center, add it to the smoothing history, and mark `target detected`.
4. Display: draws crosshairs, bounding box, target center, smoothed center, detection stats, and a human-warning overlay if needed.
5. Reconnect monitor: a background thread attempts to re-open the camera if the capture is lost; after repeated failures, it writes a `debug_frames/reconnect_failed_<timestamp>.txt` marker and saves the last good frame as `last_frame_<timestamp>.jpg`.

## Safety and Human Avoidance
- The code contains a forced "human avoidance" block: if YOLO detects class `0` (person) in a frame, that frame is locked and no targeting occurs. **Do not remove or modify** this block — it's a safety requirement.

## Tuning Guides
- Lighting: Good, even lighting improves color-masking reliability. Avoid strong shadows or direct glare on the target.
- HSV tuning (color filter): The code currently uses `lower_green = [35, 60, 40]` and `upper_green = [80, 255, 255]` to detect green targets.
  - To detect a different color (e.g., blue), update the HSV bounds in `detect_target()` near the top of the function, or modify the code to use the `--h-min`/`--h-max` flags.
  - Example blue range (approx): H 100-130, Saturation 60-255, Value 40-255.
- YOLO confidence (`--conf`): Increase to reduce false positives; lower to be more permissive.
- Minimum area (`--min-area`): Increase to ignore small detections/noise; decrease to detect small distant targets.
- Smoothing (`--smooth`): Increase to get steadier aim at the cost of responsiveness.

### Quick code change to use CLI HSV flags
Inside `detect_target()` replace the hard-coded `lower_green` / `upper_green` arrays with:
```python
lower = np.array([self.h_min, 60, 40])
upper = np.array([self.h_max, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
```
Then tune `--h-min` / `--h-max` on the command line.

## Troubleshooting
- Camera won't open: verify URL or index, check network connectivity to ESP32-CAM, ensure the ESP32 web server is running and reachable from the Pi/PC.
- No detections or low lock rate: try increasing lighting, tune HSV ranges, lower `--conf` to 0.2 for testing, or reduce `--min-area`.
- Too many false positives: raise `--conf`, tighten HSV ranges (narrow H or increase S/V min), increase `--min-area`.
- Reconnect markers appear: check network stability and power to the ESP32. See `debug_frames/` for saved logger files and the last frame image when available.

## Files of Interest
- `nerf_detector.py` — main program (this README documents it).
- `yolov8n.pt` — the YOLOv8-nano model file used for inference.
- `debug_frames/` — created at runtime when reconnect problems occur (contains marker files and saved frames).

## Glossary / Definitions
- YOLOv8: A real-time object detection model (You Only Look Once, v8). Produces bounding boxes, class IDs, and confidences.
- HSV: Hue-Saturation-Value color space. Hue is what we usually mean by "color" (0–180 in OpenCV). Saturation is color intensity, Value is brightness.
- Mask: A binary image (0 or 255) where pixels inside the color range are 255 and others are 0.
- ROI: Region Of Interest — a sub-image (e.g., a YOLO bounding box) used for per-box checks.
- BBox: Bounding box; returned as (x1, y1, x2, y2) coordinates or (x, y, w, h).
- Confidence: Model's estimated probability for that detection (0.0–1.0).
- Smoothing: Averaging recent centers to provide a stable aim point.
- Human avoidance: Safety logic that disables targeting when a person is detected.

## Next Steps / Homework for Students
- Test with a physical ESP32-CAM stream and verify reconnect behavior.
- Add command-line wiring so `--h-min`/`--h-max` actually change the mask without editing code (small patch suggested above).
- Experiment with a blue target: tune HSV, `--conf`, and `--min-area` until reliable.
- Hook up the smoothed center coordinates to servo-control code and test aiming accuracy.

---
If you'd like, I can:
- Wire the CLI HSV flags into `detect_target()` so you can tune color without editing the file.
- Create a small tuning utility that shows a live HSV inspector to pick ranges interactively.

Tell me which follow-up you'd like and I'll implement it.
