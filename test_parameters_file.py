import cv2
import time

# Load ONNX model using OpenCV DNN
net = cv2.dnn.readNet("yolov5n.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Open webcam or use video
cap = cv2.VideoCapture(0)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    # Run inference
    outputs = net.forward()

    frame_count += 1
    elapsed = time.time() - start_time

    if elapsed > 10:  # Run for 10 seconds
        break

fps = frame_count / elapsed
print(f"FPS: {fps:.2f}")

cap.release()
