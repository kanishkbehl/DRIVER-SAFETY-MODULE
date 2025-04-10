# Detects:
# - Mobile phone usage using YOLOv8 COCO-pretrained model ('cell phone' class)
# - Eye closure with blink counting (MediaPipe FaceMesh)
# - Yawning detection with yawn counting (MediaPipe FaceMesh)
# - Head orientation: Left, Right, Forward, and Down (MediaPipe FaceMesh)
# - Fatigue Score (based on eye closure, blink/yawn rate, and head-down duration)
# - DANGER LEVEL classification:
#     • Critical / High / Moderate / Low based on risk factors:
#         - Eyes closed, yawning, phone detected, head down, high fatigue score
# Uses fixed camera index 1 (AVFoundation backend) with resized frame for display

import cv2
import math
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime

# Load YOLO COCO-pretrained model for phone detection
phone_model = YOLO("yolov8n.pt")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Euclidean distance
def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Fatigue score calculation
def calculate_fatigue_score(eye_closed, blink_total, yawn_total, head_direction, runtime_minutes):
    score = 0
    if eye_closed:
        score += 4
    if yawn_total > runtime_minutes * 2:
        score += 3
    if blink_total > runtime_minutes * 20:
        score += 2
    if head_direction == "Looking Down":
        score += 1
    return min(score, 10)

# Danger level classification
def get_danger_level(eye_closed, yawn, phone, head_direction, fatigue_score):
    level = "Low"
    color = (0, 255, 0)

    risk_factors = sum([
        eye_closed,
        yawn,
        phone,
        head_direction == "Looking Down",
        fatigue_score > 6
    ])

    if risk_factors >= 4:
        level = "Critical"
        color = (0, 0, 255)
    elif risk_factors == 3:
        level = "High"
        color = (0, 165, 255)
    elif risk_factors == 2:
        level = "Moderate"
        color = (0, 255, 255)
    
    return level, color

# Open camera index 1
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ Failed to open camera index 1.")
    exit()
else:
    print("✅ Using camera index 1")

# Counters
blink_total = 0
blink_closed_frames = 0
blink_threshold = 3

yawn_total = 0
yawn_open_frames = 0
yawn_threshold = 3

phone_frame_counter = 0
headpose_frame_counter = 0
distraction_status = "None"

# Runtime start
start_time = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to grab frame.")
        continue

    frame = cv2.resize(frame, (640, 360))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # YOLOv8 phone detection
    results = phone_model(frame, verbose=False)
    detections = {'phone': False}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = phone_model.names[cls].lower()
            if conf > 0.5 and 'cell phone' in label:
                detections['phone'] = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # MediaPipe FaceMesh processing
    results_mp = face_mesh.process(frame_rgb)
    eye_closed = False
    yawn = False
    head_direction = "Unknown"

    if results_mp.multi_face_landmarks:
        for face in results_mp.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face.landmark]

            # EAR
            left_eye = [33, 160, 158, 133, 153, 144]
            eye_pts = [landmarks[i] for i in left_eye]
            ear = (euclidean(eye_pts[1], eye_pts[5]) + euclidean(eye_pts[2], eye_pts[4])) / (2.0 * euclidean(eye_pts[0], eye_pts[3]))
            eye_closed = ear < 0.2

            if eye_closed:
                blink_closed_frames += 1
            else:
                if blink_closed_frames >= blink_threshold:
                    blink_total += 1
                blink_closed_frames = 0

            # MAR
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
            elif horizontal_offset > 20:
                head_direction = "Looking Right"
            elif horizontal_offset < -20:
                head_direction = "Looking Left"
            else:
                head_direction = "Looking Forward"

    # Frame counters
    if detections['phone']:
        phone_frame_counter += 1
    else:
        phone_frame_counter = max(phone_frame_counter - 1, 0)

    if head_direction in ["Looking Down", "Looking Left", "Looking Right"]:
        headpose_frame_counter += 1
    else:
        headpose_frame_counter = max(headpose_frame_counter - 1, 0)

    # Runtime + fatigue score
    runtime_minutes = (datetime.now() - start_time).total_seconds() / 60
    fatigue_score = calculate_fatigue_score(eye_closed, blink_total, yawn_total, head_direction, runtime_minutes)

    # Danger level
    danger_level, danger_color = get_danger_level(eye_closed, yawn, detections['phone'], head_direction, fatigue_score)

    # UI
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_status(label, status, y_pos):
        color = (255, 0, 0) if status else (0, 0, 255)
        text = f"{label}: {'YES' if status else 'NO'}"
        cv2.putText(frame, text, (30, y_pos), font, 0.8, color, 2)
        return y_pos + 40

    y_offset = 40
    y_offset = draw_status("Phone", detections['phone'], y_offset)
    y_offset = draw_status("Eyes Closed", eye_closed, y_offset)
    y_offset = draw_status("Yawning", yawn, y_offset)
    cv2.putText(frame, f"Head: {head_direction}", (30, y_offset), font, 0.6, (0, 100, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Blinks: {blink_total}", (30, y_offset), font, 0.6, (255, 0, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Yawns: {yawn_total}", (30, y_offset), font, 0.6, (255, 0, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Fatigue Score: {fatigue_score:.1f} / 10", (30, y_offset), font, 0.6, (255, 0, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"DANGER LEVEL: {danger_level}", (30, y_offset), font, 0.7, danger_color, 3)

    cv2.imshow("Driver Safety Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()