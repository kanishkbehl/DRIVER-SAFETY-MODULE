# 👁️‍🗨️ Driver Safety Detection System Using Raspberry Pi 5

A real-time AI-powered Driver Safety System developed on **Raspberry Pi 5**, this upgraded safety solution combines **YOLOv8 object detection** and **MediaPipe FaceMesh** to analyze critical driver behaviors. It assesses fatigue, distraction, and unsafe activities like mobile phone usage, eye closure, yawning, and abnormal head positions to alert the driver instantly and ensure road safety.

---

## 🚦 Detects:

- 📱 **Mobile phone usage** (YOLOv8 COCO-pretrained model, `cell phone` class)
- 👁️ **Eye closure** with **blink detection and counting** (MediaPipe FaceMesh)
- 😮 **Yawning detection** with **yawn counter** (MediaPipe FaceMesh)
- 🧠 **Head orientation**:
  - Looking Left
  - Looking Right
  - Looking Forward
  - Looking Down
- 💤 **Fatigue Score**:
  - Calculated using:
    - Total blinks
    - Yawn frequency
    - Eye closure duration
    - Head-down duration
- ⚠️ **DANGER LEVEL** classification:
  - 🔴 Critical
  - 🟠 High
  - 🟡 Moderate
  - 🟢 Low

---

## 🧠 Features

- Real-time video processing from external USB camera
- Fatigue scoring logic for driver awareness tracking
- Landmark-based head orientation calculation
- Real-time graphical feedback and status overlay
- GPIO alerts (LED & buzzer) for dangerous behavior
- Highly efficient and lightweight — runs on edge using Raspberry Pi 5
- Modular code for easy extension and dataset training

---

## 📦 Tech Stack

### 🔌 Hardware:

- Raspberry Pi 5
- LED & Buzzer (for alerts)
- Breadboard, jumper wires, resistors (for GPIO setup)

### 🧰 Software & Libraries:

- **Python 3**
- [`ultralytics`](https://github.com/ultralytics/ultralytics) (YOLOv8)
- [`opencv-python`](https://pypi.org/project/opencv-python/)
- [`mediapipe`](https://github.com/google/mediapipe)
- `datetime`, `math`, `numpy`
- [`gpiozero`](https://gpiozero.readthedocs.io/en/stable/)
- [`picamera2`](https://github.com/raspberrypi/picamera2) *(if using Pi camera instead of USB)*

---

## 🔍 How It Works

1. **YOLOv8** detects mobile phone usage using the `cell phone` class from the COCO-pretrained model (`yolov8n.pt`).
2. **MediaPipe FaceMesh** extracts facial landmarks for:
   - Eye aspect ratio (EAR) → Determines if eyes are closed
   - Mouth aspect ratio (MAR) → Determines yawning
   - Landmark vector relationships → Determines head orientation
3. **Fatigue Score Calculation**:
   - Based on eye closure, blink/yawn rate, head down posture over runtime
   - Score ranges from 0 (alert) to 10 (critical)
4. **Danger Level Classification**:
   - Uses binary risk factors: phone, yawning, eyes closed, head down, high fatigue
   - 5 risk factors → Critical
   - 4 → High
   - 3 → Moderate
   - ≤2 → Low
5. **Real-Time Output**:
   - Live camera feed with overlaid status (YES/NO), head orientation, fatigue score, and danger level
