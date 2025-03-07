import torch
import cv2
import pyttsx3
import requests  # Import the requests library
import numpy as np
import warnings
import time     # Import the time library

# Filter out the specific FutureWarning related to torch.cuda.amp.autocast from ultralytics
warnings.filterwarnings("ignore", category=FutureWarning)# <--- Add this line

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)

# Load YOLOv5 Nano model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')

# ESP32-S3 stream URL - Replace with your ESP32-S3 IP address
stream_url = "http://192.168.1.129/mjpeg"  

# Initialize OpenCV video capture from webcam (initially, will be replaced)
# cap = cv2.VideoCapture(0) # Original line - webcam

# Store previously announced objects
previous_objects = set()

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


    # Perform object detection
    results = model(frame)

    # Extract detected objects as a set
    detected_objects = set(results.pandas().xyxy[0]['name'].tolist())


    # Determine new objects (not previously announced)
    new_objects = detected_objects - previous_objects

    if new_objects:    # If there are new objects detected
        announcement = "I see " + ", ".join(new_objects)
        print(announcement) # Print to console
        engine.say(announcement)
        engine.runAndWait()# Speak detected objects
        previous_objects = detected_objects # Update previous objects

    # Display results on screen
    #cv2.imshow('YOLOv5 Nano - Live Detection', results.render()[0])

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release() # No webcam capture anymore - not using webcam directly
cv2.destroyAllWindows()