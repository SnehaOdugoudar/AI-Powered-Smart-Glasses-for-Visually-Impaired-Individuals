import torch
import cv2
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv5 Nano model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# Store previously announced objects
previous_objects = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Extract detected objects as a set
    detected_objects = set(results.pandas().xyxy[0]['name'].tolist())

    # Determine new objects (not previously announced)
    new_objects = detected_objects - previous_objects

    if new_objects:  # If there are new objects detected
        announcement = "I see " + ", ".join(new_objects)
        print(announcement)  # Print to console
        engine.say(announcement)
        engine.runAndWait()  # Speak detected objects
        previous_objects = detected_objects  # Update previous objects

    # Display results on screen
    cv2.imshow('YOLOv5 Nano - Live Detection', results.render()[0])

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


