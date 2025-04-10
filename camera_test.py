import cv2

for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ Camera index {i} is working")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)
            cap.release()
            cv2.destroyAllWindows()
            break
    else:
        print(f"❌ Camera index {i} failed")
