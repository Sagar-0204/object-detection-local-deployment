import cv2
from ultralytics import YOLO
import time

model = YOLO("yolov8n.onnx", task="detect")

cv2.setUseOptimized(True)
cv2.setNumThreads(16)

cap = cv2.VideoCapture(0)

cur_frame = 0
start_time = time.time_ns()

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent performance
    frame = cv2.resize(frame, (640, 640))

    results = model(frame, verbose=False, device="cpu")

    annotated_frame = results[0].plot()

    cur_frame += 1
    cur_time = time.time_ns()
    fps = cur_frame / (cur_time - start_time) * 10**9

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
