import torch
import cv2

# Load YOLOv5 Nano model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt', force_reload=True)

# Open webcam (0 for default camera, or replace with a video file)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Show results
    cv2.imshow('YOLOv5 Nano - Live Detection', results.render()[0])

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
