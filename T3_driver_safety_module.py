# Detects:
# - Helmet usage (YOLOv8)
# - Mobile phone usage (YOLOv8)
# - Eye closure with blink counting (MediaPipe FaceMesh)
# - Yawning detection with yawn counting (MediaPipe FaceMesh)
# - Head orientation: Left, Right, or Forward (MediaPipe FaceMesh)
# Uses fixed camera index 1 (AVFoundation backend)

import cv2
import math
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
helmet_phone_model = YOLO("/Users/kanishkbehl/Desktop/DRIVER SAFETY MODULE/newhelemt_phone.pt")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Euclidean distance
def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Open known working camera (index 1)
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ Failed to open camera index 1.")
    exit()
else:
    print("✅ Using camera index 1")

# Blink & yawn counters
blink_total = 0
blink_closed_frames = 0
blink_threshold = 3

yawn_total = 0
yawn_open_frames = 0
yawn_threshold = 3

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to grab frame.")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # YOLO detection
    results = helmet_phone_model(frame, verbose=False)
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

    # MediaPipe detection
    results_mp = face_mesh.process(frame_rgb)
    eye_closed = False
    yawn = False
    head_direction = "Unknown"

    if results_mp.multi_face_landmarks:
        for face in results_mp.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face.landmark]

            # Eye Aspect Ratio (EAR)
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

            # Mouth Aspect Ratio (Yawn detection)
            mar = euclidean(landmarks[13], landmarks[14]) / euclidean(landmarks[61], landmarks[291])
            yawn = mar > 0.5

            if yawn:
                yawn_open_frames += 1
            else:
                if yawn_open_frames >= yawn_threshold:
                    yawn_total += 1
                yawn_open_frames = 0

            # Head orientation
            nose = landmarks[1]
            left_eye_center = np.mean([landmarks[33], landmarks[133]], axis=0)
            right_eye_center = np.mean([landmarks[362], landmarks[263]], axis=0)
            eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
            offset = nose[0] - eye_center_x

            if offset > 20:
                head_direction = "Looking Right"
            elif offset < -20:
                head_direction = "Looking Left"
            else:
                head_direction = "Looking Forward"

    # Status Display
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_status(label, status, y_pos):
        color = (255, 0, 0) if status else (0, 0, 255)
        text = f"{label}: {'YES' if status else 'NO'}"
        cv2.putText(frame, text, (30, y_pos), font, 0.8, color, 2)
        return y_pos + 40

    y_offset = 40
    y_offset = draw_status("Helmet", detections['helmet'], y_offset)
    y_offset = draw_status("Phone", detections['phone'], y_offset)
    y_offset = draw_status("Eyes Closed", eye_closed, y_offset)
    y_offset = draw_status("Yawning", yawn, y_offset)
    cv2.putText(frame, f"Head: {head_direction}", (30, y_offset), font, 0.8, (0, 100, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Blinks: {blink_total}", (30, y_offset), font, 0.8, (255, 0, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"Yawns: {yawn_total}", (30, y_offset), font, 0.8, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Driver Safety Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
