# Detects:
# - Mobile phone usage using YOLOv8 COCO-pretrained model ('cell phone' class)
# - Draws red bounding box and confidence score if phone is detected (confidence > 0.4)
# - Simple real-time object detection from webcam feed (camera index 1)

from ultralytics import YOLO
import cv2

# Load pre-trained COCO model
model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt for better accuracy

# Start camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = r.conf[0]
        cls = int(r.cls[0])
        label = model.names[cls]

        # Check if it's a phone
        if label == "cell phone" and conf > 0.4:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Phone Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
