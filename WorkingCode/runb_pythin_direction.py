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

# Define minimum bounding box area for announcement (tune this value)
MIN_BBOX_AREA = 5000  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame width
    frame_width = frame.shape[1]

    # Perform object detection
    results = model(frame)
    
    # Extract detected objects and their positions
    detected_objects = set()
    
    for index, row in results.pandas().xyxy[0].iterrows():
        x_min, x_max, y_min, y_max, obj_name = row['xmin'], row['xmax'], row['ymin'], row['ymax'], row['name']
        bbox_area = (x_max - x_min) * (y_max - y_min)  # Calculate bounding box area
        x_center = (x_min + x_max) / 2  # Calculate object’s center position

        if bbox_area > MIN_BBOX_AREA:  # Only consider close objects
            # Determine object location
            if x_center < frame_width / 3:
                position = "on the left"
            elif x_center > 2 * frame_width / 3:
                position = "on the right"
            else:
                position = "in the center"

            detected_objects.add(f"{obj_name} {position}")

    # Determine new close objects (not previously announced)
    new_objects = detected_objects - previous_objects

    if new_objects:  # If there are new close objects detected
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
