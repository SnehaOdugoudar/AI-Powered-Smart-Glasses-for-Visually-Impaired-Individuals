import torch
import cv2
import pyttsx3
import requests  # Import the requests library
import numpy as np
import io
import time     # Import the time library

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)

# Load YOLOv5 Nano model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')

# ESP32-S3 stream URL - Replace with your ESP32-S3 IP address
stream_url = "http://192.168.1.129/mjpeg"  # <--- REPLACE THIS WITH YOUR ESP32-S3 IP ADDRESS

# Initialize OpenCV video capture from webcam (initially, will be replaced)
# cap = cv2.VideoCapture(0) # Original line - webcam

# Store previously announced objects
previous_objects = set()
MIN_BBOX_AREA = 5000
LARGE_BBOX_AREA = 15000

while True:
    try:
        response = requests.get(stream_url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]  # Extract JPEG frame
                bytes_data = bytes_data[b+2:]  # Remove processed frame

                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    break # Frame decoded successfully, exit chunk loop
    except requests.exceptions.RequestException as e:
        print(f"Error fetching frame: {e}")
        time.sleep(1) # Wait and retry
        continue # Go to the next iteration of the while loop

    if frame is None:
        print("Error decoding frame")
        time.sleep(1) # Wait and retry
        continue # Go to the next iteration of the while loop


    frame_width = frame.shape[1]
    results = model(frame)

    detected_objects = set()

    for index, row in results.pandas().xyxy[0].iterrows():
        x_min, x_max, y_min, y_max, obj_name = row['xmin'], row['xmax'], row['ymin'], row['ymax'], row['name']
        bbox_area = (x_max - x_min) * (y_max - y_min)
        x_center = (x_min + x_max) / 2

        if bbox_area > MIN_BBOX_AREA:
            if x_center < frame_width / 3:
                position = "on the left"
            elif x_center > 2 * frame_width / 3:
                position = "on the right"
            else:
                position = "in the center"

            if bbox_area > LARGE_BBOX_AREA:
                detected_objects.add(f"WARNING! {obj_name} VERY CLOSE {position}")
                engine.setProperty('volume', 1.0)
            else:
                detected_objects.add(f"{obj_name} {position}")
                engine.setProperty('volume', 0.7)

    new_objects = detected_objects - previous_objects

    if new_objects:
        announcement = "I see " + ", ".join(new_objects)
        print(announcement)
        engine.say(announcement)
        engine.runAndWait()
        previous_objects = detected_objects

    # Display results - comment out if running purely via SSH without desktop
    cv2.imshow('YOLOv5 Nano - Live Detection', results.render()[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release() # No webcam capture anymore
cv2.destroyAllWindows() # Comment out if running purely via SSH without desktop