# Detects:
# - Eye closure and yawning using custom YOLOv8 model (trained on eyes/yawn classes)
# - Performs real-time detection on webcam feed with resized frames for faster inference
# - Prints detected class names to console
# - Displays annotated video with bounding boxes and class labels
# Designed to run on local machine (adjust camera index/path as needed)

# -*- coding: utf-8 -*-
"""Eyes_yawn_detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ODO7kgqQ6HzmVchjp7p-cmOdMWF5VMs2
"""

import cv2
from ultralytics import YOLO

# Load YOLO model (Update path as needed)
model = YOLO(r"F:\Intute.ai\yoloeyesyawn\best.pt")

# Open webcam (0 for default camera, change if using external camera)
cap = cv2.VideoCapture(0)

# Reduce input frame size for smoother detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not capture frame.")
        break

    # Resize image for faster inference
    small_frame = cv2.resize(frame, (320, 240))

    # Run YOLO inference
    results = model(small_frame)

    # Extract and print detected class names
    detected_classes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class index

            # Handle missing class names gracefully
            class_name = model.names.get(class_id, f"Unknown({class_id})")
            detected_classes.append(class_name)

    if detected_classes:
        print(f"Detected: {', '.join(detected_classes)}")

    # Draw bounding boxes on the original frame (not resized)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Eye & Yawn Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()