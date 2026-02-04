# Autonomous Turret Track: Nerf-Bullseye-Seg

## Overview
Autonomy pushes engineering beyond direct control and into the realm of independent decision-making. This project transforms a standard Nerf platform into a **self-directed autonomous system**.

By integrating **Computer Vision** (YOLOv8 + SAM 2.1) with precise hardware control, the turret can:
- Sense its environment  
- Interpret visual data  
- Independently track and neutralize targets  

—all without human intervention.

---

## Objectives
- **Computer Vision Integration**  
  Real-time target detection and segmentation with a live camera feed displayed via an LED or monitor interface.

- **Fully Autonomous Logic**  
  A static or mobile system capable of identifying, tracking, and firing at a target autonomously.

- **Engineering Presentation**  
  A documented journey of system design, functionality, and iterative problem-solving.

---

## Constraints and Compliance
- **Dual-Axis Aiming**  
  Precise movement across at least two axes (Pitch and Yaw).

- **Autonomous Firing**  
  Launch sequence triggered independently based on CV “lock-on.”

- **Non-Invasive Modification**  
  Strictly follows the *no-modifications* rule — the Nerf gun is fired using **external actuators only**, with no changes to the original firing mechanism or pellets.

- **Durability**  
  Built to withstand repeated testing and high-vibration firing sequences.

---

## Tech Stack
- **Detection:** YOLOv8 (real-time object detection)  
- **Segmentation:** SAM 2.1 (Segment Anything Model)  
- **Safety Logic:** Integrated human-avoidance protocols (skips processing if `cls == 0`)  
- **Language:** Python 3.11+

---

## Repository Structure
```plaintext
nerf-bullseye-seg/
├── nerf_detector.py      # Core CV detection and tracking logic
├── Instructions.md       # Hardware setup and calibration guide
├── requirements.txt      # Python dependencies
└── .gitignore            # Excludes heavy model weights (.pt files)

## Created By 
**BU Women in Data Science (WiDS)**

**Connect With Us**
- **Instagram:** @bu_wids
- **Email:** bostonuwids@gmail.com
 


