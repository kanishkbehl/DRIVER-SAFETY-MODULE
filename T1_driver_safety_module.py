# Detects:
# - Helmet usage (YOLOv8)
# - Mobile phone usage (YOLOv8)
# - Eye closure (MediaPipe FaceMesh)
# - Yawning (MediaPipe FaceMesh)
# Also includes auto camera detection (indexes 0 to 2)

import cv2
import math
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

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
                    cv2.putText(frame, f"Helmet {conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif 'phone' in label:
                    detections['phone'] = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # MediaPipe detection
    results_mp = face_mesh.process(frame_rgb)
    eye_closed = False
    yawn = False

    if results_mp.multi_face_landmarks:
        for face in results_mp.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face.landmark]

            # Eye closure
            left_eye = [33, 160, 158, 133, 153, 144]
            eye_pts = [landmarks[i] for i in left_eye]
            ear = (euclidean(eye_pts[1], eye_pts[5]) + euclidean(eye_pts[2], eye_pts[4])) / (2.0 * euclidean(eye_pts[0], eye_pts[3]))
            if ear < 0.2:
                eye_closed = True

            # Yawn detection
            mar = euclidean(landmarks[13], landmarks[14]) / euclidean(landmarks[61], landmarks[291])
            if mar > 0.5:
                yawn = True

    # Status display
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_status(label, status, y_pos):
        color = (255, 0, 0) if status else (0, 0, 255)  # Blue if safe, Red if violation
        text = f"{label}: {'YES' if status else 'NO'}"
        cv2.putText(frame, text, (30, y_pos), font, 0.8, color, 2)
        return y_pos + 40

    y_offset = 50
    y_offset = draw_status("Helmet", detections['helmet'], y_offset)
    y_offset = draw_status("Phone", detections['phone'], y_offset)
    y_offset = draw_status("Eyes Closed", eye_closed, y_offset)
    y_offset = draw_status("Yawning", yawn, y_offset)

    # Show frame
    cv2.imshow("Driver Safety Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import math
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load YOLO model
helmet_phone_model = YOLO("/Users/kanishkbehl/Desktop/DRIVER SAFETY MODULE/newhelemt_phone.pt")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Euclidean distance
def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Try opening from multiple camera indices
def get_camera():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera index {i} selected")
                return cap
            cap.release()
    print("❌ Could not find any usable camera.")
    exit()

cap = get_camera()

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
                    cv2.putText(frame, f"Helmet {conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif 'phone' in label:
                    detections['phone'] = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # MediaPipe detection
    results_mp = face_mesh.process(frame_rgb)
    eye_closed = False
    yawn = False

    if results_mp.multi_face_landmarks:
        for face in results_mp.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face.landmark]

            # Eye closure
            left_eye = [33, 160, 158, 133, 153, 144]
            eye_pts = [landmarks[i] for i in left_eye]
            ear = (euclidean(eye_pts[1], eye_pts[5]) + euclidean(eye_pts[2], eye_pts[4])) / (2.0 * euclidean(eye_pts[0], eye_pts[3]))
            if ear < 0.2:
                eye_closed = True

            # Yawn detection
            mar = euclidean(landmarks[13], landmarks[14]) / euclidean(landmarks[61], landmarks[291])
            if mar > 0.5:
                yawn = True

    # Status display
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_status(label, status, y_pos):
        color = (255, 0, 0) if status else (0, 0, 255)
        text = f"{label}: {'YES' if status else 'NO'}"
        cv2.putText(frame, text, (30, y_pos), font, 0.8, color, 2)
        return y_pos + 40

    y_offset = 50
    y_offset = draw_status("Helmet", detections['helmet'], y_offset)
    y_offset = draw_status("Phone", detections['phone'], y_offset)
    y_offset = draw_status("Eyes Closed", eye_closed, y_offset)
    y_offset = draw_status("Yawning", yawn, y_offset)

    cv2.imshow("Driver Safety Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
