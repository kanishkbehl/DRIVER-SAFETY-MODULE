# Detects:
# - Helmet usage (YOLOv8 custom model)
# - Mobile phone usage (YOLOv8 custom model)
# - Eye closure with blink counting (MediaPipe FaceMesh)
# - Yawning detection with yawn counting (MediaPipe FaceMesh)
# - Head orientation: Left, Right, Forward, and Down (MediaPipe FaceMesh)
# - Fatigue Score (based on % eye closure, blink/yawn rate, head-down %)
# - Real-time fatigue score chart embedded in OpenCV feed using Matplotlib
# Uses fixed camera index 1 (AVFoundation backend)

import cv2
import math
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from matplotlib.figure import Figure
import io

# Load YOLO model
helmet_phone_model = YOLO("/Users/kanishkbehl/Desktop/DRIVER SAFETY MODULE/newhelemt_phone.pt")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Euclidean distance function
def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Fatigue score calculation
def calculate_fatigue_score(closed_frames, total_frames, blink_rate, yawn_rate, head_down_frames):
    eye_closed_pct = closed_frames / total_frames if total_frames > 0 else 0
    head_down_pct = head_down_frames / total_frames if total_frames > 0 else 0

    score = 0
    if eye_closed_pct > 0.3:
        score += 4
    elif eye_closed_pct > 0.15:
        score += 2

    if yawn_rate > 4:
        score += 3
    elif yawn_rate > 2:
        score += 2
    elif yawn_rate > 1:
        score += 1

    if blink_rate > 25:
        score += 2
    elif blink_rate > 20:
        score += 1

    if head_down_pct > 0.3:
        score += 1

    return min(score, 10)

# Initialize chart data
timestamps = []
fatigue_scores = []
chart_update_interval = 5
last_chart_update = datetime.now()

# Start camera
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ Failed to open camera index 1.")
    exit()
else:
    print("✅ Using camera index 1")

blink_total = 0
blink_closed_frames = 0
blink_threshold = 3
closed_frames = 0

yawn_total = 0
yawn_open_frames = 0
yawn_threshold = 3

head_down_frames = 0
total_frames = 0

start_time = datetime.now()

# Matplotlib chart setup (for embedding into OpenCV)
fig = Figure(figsize=(4, 2))
ax = fig.add_subplot(111)
canvas = agg.FigureCanvasAgg(fig)  # Initialized once

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to grab frame.")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = helmet_phone_model.predict(source=frame, verbose=False, stream=False)
    detections = {'helmet': False, 'phone': False}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = helmet_phone_model.names[cls].lower()
            if conf > 0.5:
                if 'helmet' in label:
                    detections['helmet'] = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                elif 'phone' in label:
                    detections['phone'] = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    results_mp = face_mesh.process(frame_rgb)
    eye_closed = False
    yawn = False
    head_direction = "Unknown"

    total_frames += 1

    if results_mp.multi_face_landmarks:
        for face in results_mp.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face.landmark]

            # Eye
            left_eye = [33, 160, 158, 133, 153, 144]
            eye_pts = [landmarks[i] for i in left_eye]
            ear = (euclidean(eye_pts[1], eye_pts[5]) + euclidean(eye_pts[2], eye_pts[4])) / (2.0 * euclidean(eye_pts[0], eye_pts[3]))
            eye_closed = ear < 0.2

            if eye_closed:
                blink_closed_frames += 1
                closed_frames += 1
            else:
                if blink_closed_frames >= blink_threshold:
                    blink_total += 1
                blink_closed_frames = 0

            # Yawn
            mar = euclidean(landmarks[13], landmarks[14]) / euclidean(landmarks[61], landmarks[291])
            yawn = mar > 0.5

            if yawn:
                yawn_open_frames += 1
            else:
                if yawn_open_frames >= yawn_threshold:
                    yawn_total += 1
                yawn_open_frames = 0

            # Head orientation
            nose_tip = landmarks[1]
            left_eye_center = np.mean([landmarks[33], landmarks[133]], axis=0)
            right_eye_center = np.mean([landmarks[362], landmarks[263]], axis=0)
            chin = landmarks[152]

            eyes_mid_y = (left_eye_center[1] + right_eye_center[1]) / 2
            eyes_to_chin = chin[1] - eyes_mid_y
            nose_to_eyes = nose_tip[1] - eyes_mid_y
            down_ratio = nose_to_eyes / eyes_to_chin
            horizontal_offset = nose_tip[0] - ((left_eye_center[0] + right_eye_center[0]) / 2)

            if down_ratio > 0.6:
                head_direction = "Looking Down"
                head_down_frames += 1
            elif horizontal_offset > 20:
                head_direction = "Looking Right"
            elif horizontal_offset < -20:
                head_direction = "Looking Left"
            else:
                head_direction = "Looking Forward"

    runtime_minutes = (datetime.now() - start_time).total_seconds() / 60
    blink_rate = blink_total / runtime_minutes if runtime_minutes > 0 else 0
    yawn_rate = yawn_total / runtime_minutes if runtime_minutes > 0 else 0

    fatigue_score = calculate_fatigue_score(closed_frames, total_frames, blink_rate, yawn_rate, head_down_frames)

    now = datetime.now()
    if (now - last_chart_update).total_seconds() >= chart_update_interval:
        timestamps.append(now)
        fatigue_scores.append(fatigue_score)
        last_chart_update = now

    # Draw chart image efficiently
    ax.clear()
    ax.set_ylim(0, 10)
    ax.set_title("Fatigue Score Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    if timestamps:
        x = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        ax.plot(x, fatigue_scores, color='blue')
    canvas.draw()
    buf = canvas.buffer_rgba()
    plot_img = np.asarray(buf)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    plot_img = cv2.resize(plot_img, (400, 200))
    frame[0:200, -400:] = plot_img

    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_status(label, status, y_pos):
        color = (255, 0, 0) if status else (0, 0, 255)
        text = f"{label}: {'YES' if status else 'NO'}"
        cv2.putText(frame, text, (30, y_pos), font, 0.8, color, 2)
        return y_pos + 40

    y_offset = 220
    y_offset = draw_status("Helmet", detections['helmet'], y_offset)
    y_offset = draw_status("Phone", detections['phone'], y_offset)
    y_offset = draw_status("Eyes Closed", eye_closed, y_offset)
    y_offset = draw_status("Yawning", yawn, y_offset)
    cv2.putText(frame, f"Head: {head_direction}", (30, y_offset), font, 0.8, (0, 100, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Blinks: {blink_total}", (30, y_offset), font, 0.8, (255, 0, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Yawns: {yawn_total}", (30, y_offset), font, 0.8, (255, 0, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Fatigue Score: {fatigue_score:.1f} / 10", (30, y_offset), font, 0.8, (255, 0, 0), 2)

    cv2.imshow("Driver Safety Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
