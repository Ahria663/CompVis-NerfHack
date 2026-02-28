# Nerf Turret API Setup Instructions

## Overview

This system has two parts:

1. **Server (your computer)** - Runs YOLO model, does the heavy processing
2. **Client (Raspberry Pi Zero)** - Detects motion, sends frames to server, controls servos

**Answer to your question:** Yes! The Pi does NOT need ultralytics, YOLO, or heavy ML packages. It only needs lightweight packages (`requests`, `opencv`, `numpy`). All the AI processing happens on your computer.

---

## Part 1: Server Setup (Your Computer)

### Every Time You Want to Run the Server:

#### Step 1: Open Git Bash and navigate to your project

```bash
cd the directory that contains your nerf project
```

#### Step 2: Create & Activate your Python 3.10 environment
#### PS: You need Python 3.10 installed

**Mac/Linux:**
```bash
python3.10 -m venv venv310
source venv310/bin/activate
```

**Windows CMD:**
```cmd
py -3.10 -m venv venv310
venv310\Scripts\activate
```

**Windows PowerShell:**
```powershell
py -3.10 -m venv venv310
.\venv310\Scripts\Activate.ps1
```

You should see `(venv310)` at the start of your prompt.

#### Step 3: Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
Loading YOLOv8 model...
✓ Model loaded
```

**Keep this terminal open!** The server must stay running.

---

### Testing the Server (Optional but Recommended)

Open a NEW Git Bash terminal (keep server running in the first one):

#### Test 1: Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","model":"yolov8n"}
```

#### Test 2: Test with an image

```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/detect
```

Expected response (something like):
```json
{"found":true,"center":[1862,1251],"bbox":[1425,394,875,1714],"confidence":0.52,"area":1499750,"human_detected":false,"frame_center":[1843,1229]}
```

---

### Get Your Computer's IP Address

In Git Bash or Command Prompt:

```bash
ipconfig
```

Look for something like:
```
Wireless LAN adapter Wi-Fi:
   IPv4 Address. . . . . . . . . . . : 192.168.1.105
```

**Write this IP down!** You'll need it for the Pi.

---

## Part 2: Raspberry Pi Zero Setup

### One-Time Setup (Do This Once)

#### Step 1: Connect Pi to same WiFi network as your computer

#### Step 2: SSH into your Pi (from your computer)

```bash
ssh pi@raspberrypi.local
```

Or use the Pi's IP address:
```bash
ssh pi@192.168.1.XXX
```

Default password is usually `raspberry`

#### Step 3: Update the Pi

```bash
sudo apt update && sudo apt upgrade -y
```

#### Step 4: Install required packages

```bash
sudo apt install -y python3-opencv python3-pip python3-numpy
pip3 install requests
```

**That's it!** The Pi only needs these lightweight packages:
- `opencv` - for camera and motion detection
- `numpy` - for array operations  
- `requests` - for sending HTTP requests to your server

NO ultralytics, NO YOLO, NO heavy ML packages needed on Pi!

#### Step 5: Copy pi_client.py to the Pi

From your computer (in a new terminal):

```bash
scp pi_client.py pi@raspberrypi.local:~/pi_client.py
```

Or create the file directly on the Pi:

```bash
nano ~/pi_client.py
```

Then paste the pi_client.py code and save (Ctrl+X, Y, Enter).

#### Step 6: Edit the API_URL in pi_client.py

```bash
nano ~/pi_client.py
```

Change this line:
```python
API_URL = "http://YOUR_SERVER_IP:8000/detect"
```

To your computer's IP:
```python
API_URL = "http://192.168.1.105:8000/detect"  # Use YOUR IP
```

Save and exit (Ctrl+X, Y, Enter).

---

### Running the Pi Client (Every Time)

#### Step 1: Make sure your server is running on your computer first!

#### Step 2: SSH into the Pi

```bash
ssh pi@raspberrypi.local
```

#### Step 3: Run the client

```bash
python3 ~/pi_client.py
```

You should see:
```
==================================================
NERF TURRET - PI CLIENT
==================================================
API: http://192.168.1.105:8000/detect
Camera: 0
==================================================
✓ Camera ready
✓ Monitoring for motion...
Press Ctrl+C to stop
```

When motion is detected, it will call your server and print results:
```
TARGET: [320, 240], offset: (10, -5)
```

---

## Quick Reference Card

### On Your Computer (Server)

```bash
# Navigate to project
cd to-your-project-directory

# Activate environment
source venv310/Scripts/activate

# Run server
uvicorn main_api:app --host 0.0.0.0 --port 8000
```

### On Raspberry Pi (Client)

```bash
# SSH into Pi
ssh pi@raspberrypi.local

# Run client
python3 ~/pi_client.py
```

---

## Troubleshooting

### "Connection refused" on Pi

- Make sure server is running on your computer
- Check that both devices are on same WiFi network
- Verify the IP address is correct
- Check Windows Firewall isn't blocking port 8000:
  - Windows Security → Firewall → Allow an app → Allow Python

### "Cannot open camera" on Pi

- Check camera is connected: `ls /dev/video*`
- Enable camera in Pi settings: `sudo raspi-config` → Interface Options → Camera
- Try `CAMERA_SOURCE = 1` instead of `0` in pi_client.py

### Server not starting

- Make sure you activated the venv310 environment
- Check for syntax errors in main_api.py (no stray ``` backticks)
- Make sure port 8000 isn't already in use

### Slow response times

- Lower JPEG_QUALITY in pi_client.py (e.g., 60 instead of 80)
- Increase COOLDOWN_SECONDS to reduce API call frequency
- Make sure you're on 5GHz WiFi if available

---

## File Locations Summary

### On Your Computer

```
nerf-bullseye-seg/nerf-api/
├── venv310/              # Python 3.10 environment
├── main.py           # FastAPI server code
├── requirements.txt      # Server dependencies
└── yolov8n.pt           # YOLO model (auto-downloads)
```

### On Raspberry Pi

```
~/
└── pi_client.py          # Client code (motion detection + API calls)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR COMPUTER (Server)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  FastAPI + YOLO Model                                    │    │
│  │  - Receives images                                       │    │
│  │  - Runs object detection                                 │    │
│  │  - Returns target coordinates                            │    │
│  │  - Checks for humans (safety)                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ▲                                   │
│                              │ HTTP (port 8000)                  │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                         WiFi Network
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                              ▼                                   │
│                   RASPBERRY PI ZERO (Client)                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  pi_client.py                                            │    │
│  │  - Captures camera frames                                │    │
│  │  - Detects motion (lightweight)                          │    │
│  │  - Sends frame to server when motion detected            │    │
│  │  - Receives target coordinates                           │    │
│  │  - Controls servos (TODO: add your code)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│                     Camera + Servos                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Architecture?

| Task | Pi Zero | Your Computer |
|------|---------|---------------|
| Motion detection | ✅ (lightweight) | |
| YOLO inference | ❌ (too slow) | ✅ (fast) |
| Camera capture | ✅ | |
| Servo control | ✅ | |
| Heavy ML packages | ❌ (not needed!) | ✅ |

The Pi Zero only does simple tasks (motion detection, camera, servos) while your computer does the heavy AI processing. This is much faster and more practical than trying to run YOLO on a Pi Zero!
