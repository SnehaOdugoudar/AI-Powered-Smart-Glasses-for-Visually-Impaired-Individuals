# import torch
# import cv2
# import pyttsx3
# import requests  # Import the requests library
# import numpy as np
# import io
# import time     # Import the time library

# # Initialize text-to-speech engine
# engine = pyttsx3.init()
# engine.setProperty('rate', 180)

# # Load YOLOv5 Nano model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')

# # ESP32-S3 stream URL - Replace with your ESP32-S3 IP address
# stream_url = "http://192.168.1.129/mjpeg"  # <--- REPLACE THIS WITH YOUR ESP32-S3 IP ADDRESS

# # Initialize OpenCV video capture from webcam (initially, will be replaced)
# # cap = cv2.VideoCapture(0) # Original line - webcam

# # Store previously announced objects
# previous_objects = set()
# MIN_BBOX_AREA = 5000
# LARGE_BBOX_AREA = 15000

# while True:
#     try:
#         response = requests.get(stream_url, stream=True)
#         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

#         bytes_data = bytes()
#         for chunk in response.iter_content(chunk_size=1024):
#             bytes_data += chunk
#             a = bytes_data.find(b'\xff\xd8')  # JPEG start
#             b = bytes_data.find(b'\xff\xd9')  # JPEG end
#             if a != -1 and b != -1:
#                 jpg = bytes_data[a:b+2]  # Extract JPEG frame
#                 bytes_data = bytes_data[b+2:]  # Remove processed frame

#                 frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#                 if frame is not None:
#                     break # Frame decoded successfully, exit chunk loop
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching frame: {e}")
#         time.sleep(1) # Wait and retry
#         continue # Go to the next iteration of the while loop

#     if frame is None:
#         print("Error decoding frame")
#         time.sleep(1) # Wait and retry
#         continue # Go to the next iteration of the while loop


#     frame_width = frame.shape[1]
#     results = model(frame)

#     detected_objects = set()

#     for index, row in results.pandas().xyxy[0].iterrows():
#         x_min, x_max, y_min, y_max, obj_name = row['xmin'], row['xmax'], row['ymin'], row['ymax'], row['name']
#         bbox_area = (x_max - x_min) * (y_max - y_min)
#         x_center = (x_min + x_max) / 2

#         if bbox_area > MIN_BBOX_AREA:
#             if x_center < frame_width / 3:
#                 position = "on the left"
#             elif x_center > 2 * frame_width / 3:
#                 position = "on the right"
#             else:
#                 position = "in the center"

#             if bbox_area > LARGE_BBOX_AREA:
#                 detected_objects.add(f"WARNING! {obj_name} VERY CLOSE {position}")
#                 engine.setProperty('volume', 1.0)
#             else:
#                 detected_objects.add(f"{obj_name} {position}")
#                 engine.setProperty('volume', 0.7)

#     new_objects = detected_objects - previous_objects

#     if new_objects:
#         announcement = "I see " + ", ".join(new_objects)
#         print(announcement)
#         engine.say(announcement)
#         engine.runAndWait()
#         previous_objects = detected_objects

#     # Display results - comment out if running purely via SSH without desktop
#     cv2.imshow('YOLOv5 Nano - Live Detection', results.render()[0])

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # cap.release() # No webcam capture anymore
# cv2.destroyAllWindows() # Comment out if running purely via SSH without desktop


#################################
#WORKING VERSION _ DEBUG

# import torch
# import cv2
# import numpy as np
# import requests
# import time

# # ─ CONFIG ───────────────────────────────────────────────────────────────────────
# USE_WEBCAM = False
# STREAM_URL = "http://192.168.1.129/mjpeg"
# YOLO_MODEL_PATH = 'yolov5n.pt'
# YOLO_CONF = 0.1     # low to catch anything
# YOLO_IOU  = 0.45

# # ─ INIT ─────────────────────────────────────────────────────────────────────────
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
# model.conf = YOLO_CONF
# model.iou  = YOLO_IOU

# if USE_WEBCAM:
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened(): raise RuntimeError("Webcam failed")

# # ─ LOOP ─────────────────────────────────────────────────────────────────────────
# while True:
#     # acquire frame
#     if USE_WEBCAM:
#         ret, frame = cap.read()
#         if not ret: break
#     else:
#         # pull one JPEG
#         try:
#             resp = requests.get(STREAM_URL, stream=True, timeout=5)
#             buf = bytes(); frame = None
#             for chunk in resp.iter_content(1024):
#                 buf += chunk
#                 a = buf.find(b'\xff\xd8'); b = buf.find(b'\xff\xd9')
#                 if a!=-1 and b!=-1:
#                     jpg = buf[a:b+2]; buf = buf[b+2:]
#                     frame = cv2.imdecode(np.frombuffer(jpg,np.uint8), cv2.IMREAD_COLOR)
#                     break
#             if frame is None: continue
#         except:
#             time.sleep(1); continue

#     # show raw
#     cv2.imshow('RAW', frame)

#     # run YOLO
#     results = model(frame, size=320)
#     dets = results.xyxy[0].cpu().numpy()

#     # draw every box
#     for *box, conf, cls in dets:
#         x1,y1,x2,y2 = map(int, box)
#         name        = model.names[int(cls)]
#         label       = f"{name} {conf:.2f}"
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
#         cv2.putText(frame, label, (x1, y1-5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

#     cv2.imshow('DEBUG DETECTIONS', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ─ CLEAN UP ────────────────────────────────────────────────────────────────────
# if USE_WEBCAM: cap.release()
# cv2.destroyAllWindows()

####################################
#Good gojng - there is live stream

# import torch
# import cv2
# import pyttsx3
# import requests
# import numpy as np
# import time

# # ─── CONFIG ─────────────────────────────────────────────────────────────────────
# USE_WEBCAM        = False
# STREAM_URL        = "http://192.168.1.129/mjpeg"  # ← your ESP32-S3 IP
# MIN_AREA_RATIO    = 0.02   # ignore detections <2% of frame
# LARGE_AREA_RATIO  = 0.06   # mark “VERY CLOSE” if >6% of frame
# YOLO_INFERENCE_SZ = 320    # let model internally resize to 320×320
# YOLO_CONF         = 0.25
# YOLO_IOU          = 0.45
# YOLO_MODEL_PATH   = 'yolov5n.pt'
# TTS_RATE          = 180
# VOLUME_NEAR       = 1.0
# VOLUME_FAR        = 0.7

# # ─── INITIALIZE ─────────────────────────────────────────────────────────────────
# # Text-to-speech
# engine = pyttsx3.init()
# engine.setProperty('rate', TTS_RATE)

# # Load your custom YOLOv5 Nano
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
# model.conf = YOLO_CONF
# model.iou  = YOLO_IOU

# previous_objects = set()

# # Open local webcam if requested
# if USE_WEBCAM:
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open local webcam")

# # ─── MAIN LOOP ───────────────────────────────────────────────────────────────────
# while True:
#     # 1) Grab a frame
#     if USE_WEBCAM:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("[ERROR] Webcam frame grab failed")
#             break
#     else:
#         try:
#             resp = requests.get(STREAM_URL, stream=True, timeout=5)
#             resp.raise_for_status()
#             buf, frame = bytes(), None
#             for chunk in resp.iter_content(1024):
#                 buf += chunk
#                 a = buf.find(b'\xff\xd8'); b = buf.find(b'\xff\xd9')
#                 if a != -1 and b != -1:
#                     jpg   = buf[a:b+2]
#                     buf   = buf[b+2:]
#                     frame = cv2.imdecode(np.frombuffer(jpg,np.uint8), cv2.IMREAD_COLOR)
#                     break
#             if frame is None:
#                 print("[WARN] failed to decode MJPEG frame")
#                 time.sleep(0.1)
#                 continue
#         except Exception as e:
#             print(f"[WARN] stream error: {e}")
#             time.sleep(1)
#             continue

#     # 2) Dynamic area thresholds
#     h, w     = frame.shape[:2]
#     frame_area = w * h
#     min_area   = MIN_AREA_RATIO   * frame_area
#     large_area = LARGE_AREA_RATIO * frame_area

#     # 3) YOLOv5 inference (internal resize to YOLO_INFERENCE_SZ)
#     results = model(frame, size=YOLO_INFERENCE_SZ)
#     dets    = results.pandas().xyxy[0]  # xmin,ymin,xmax,ymax,conf,class,name

#     # 4) Filter & label
#     detected = set()
#     for _, d in dets.iterrows():
#         xmin, ymin, xmax, ymax = d['xmin'], d['ymin'], d['xmax'], d['ymax']
#         name                   = d['name']
#         area                   = (xmax - xmin) * (ymax - ymin)
#         if area < min_area:
#             continue

#         x_center = (xmin + xmax) / 2
#         if   x_center <  w/3: pos = "on the left"
#         elif x_center > 2*w/3: pos = "on the right"
#         else:                  pos = "in the center"

#         if area > large_area:
#             label = f"WARNING! {name} VERY CLOSE {pos}"
#             engine.setProperty('volume', VOLUME_NEAR)
#         else:
#             label = f"{name} {pos}"
#             engine.setProperty('volume', VOLUME_FAR)

#         detected.add(label)

#     # 5) Announce new ones
#     new = detected - previous_objects
#     if new:
#         announcement = "I see " + ", ".join(new)
#         print("[ANNOUNCE]", announcement)
#         engine.say(announcement)
#         engine.runAndWait()
#     previous_objects = detected

#     # 6) Display continuous overlay (press 'q' to quit)
#     vis = results.render()[0]
#     cv2.imshow('YOLOv5n Detection', vis)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ─── CLEAN UP ────────────────────────────────────────────────────────────────────
# if USE_WEBCAM:
#     cap.release()
# cv2.destroyAllWindows()



###########################################
import torch
import cv2
import pyttsx3
import requests
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")


# ─── CONFIG ─────────────────────────────────────────────────────────────────────
USE_WEBCAM        = False
STREAM_URL        = "http://192.168.1.129/mjpeg"  # ← your ESP32-S3 IP here
MIN_AREA_RATIO    = 0.02   # ignore detections <2% of frame
LARGE_AREA_RATIO  = 0.06   # mark “VERY CLOSE” if >6% of frame
YOLO_INFERENCE_SZ = 120    # model’s internal resize
YOLO_CONF         = 0.25
YOLO_IOU          = 0.45
YOLO_MODEL_PATH   = 'yolov5n.pt'
TTS_RATE          = 180
VOLUME_NEAR       = 1.0
VOLUME_FAR        = 0.7

# ─── INITIALIZE ─────────────────────────────────────────────────────────────────
engine = pyttsx3.init()
engine.setProperty('rate', TTS_RATE)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
model.conf = YOLO_CONF
model.iou  = YOLO_IOU

previous_objects = set()

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open local webcam")

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────────
try:
    while True:
        # 1) Acquire frame
        if USE_WEBCAM:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam frame grab failed")
                break
        else:
            try:
                resp = requests.get(STREAM_URL, stream=True, timeout=5)
                resp.raise_for_status()
                buf, frame = bytes(), None
                for chunk in resp.iter_content(1024):
                    buf += chunk
                    a = buf.find(b'\xff\xd8'); b = buf.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg   = buf[a:b+2]
                        buf   = buf[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg,np.uint8), cv2.IMREAD_COLOR)
                        break
                if frame is None:
                    time.sleep(0.1)
                    continue
            except Exception as e:
                print(f"[WARN] Stream error: {e}")
                time.sleep(1)
                continue

        # 2) Compute dynamic thresholds
        h, w      = frame.shape[:2]
        area      = w * h
        min_area  = MIN_AREA_RATIO   * area
        large_area= LARGE_AREA_RATIO * area

        # 3) YOLO inference
        results = model(frame, size=YOLO_INFERENCE_SZ)
        dets    = results.pandas().xyxy[0]  # xmin,ymin,xmax,ymax,conf,class,name

        detected = set()
        for _, d in dets.iterrows():
            xmin, ymin, xmax, ymax = d['xmin'], d['ymin'], d['xmax'], d['ymax']
            name                   = d['name']
            box_area               = (xmax - xmin) * (ymax - ymin)
            if box_area < min_area:
                continue

            x_center = (xmin + xmax) / 2
            if   x_center <  w/3: pos = "on the left"
            elif x_center > 2*w/3: pos = "on the right"
            else:                  pos = "in the center"

            if box_area > large_area:
                label = f"WARNING! {name} VERY CLOSE {pos}"
                engine.setProperty('volume', VOLUME_NEAR)
            else:
                label = f"{name} {pos}"
                engine.setProperty('volume', VOLUME_FAR)

            detected.add(label)

        # 4) Announce new detections
        new = detected - previous_objects
        if new:
            announcement = "I see " + ", ".join(new)
            print("[ANNOUNCE]", announcement)
            engine.say(announcement)
            engine.runAndWait()
        previous_objects = detected

        # tiny sleep to avoid hammering CPU/HTTP
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopping detection.")

# ─── CLEAN UP ────────────────────────────────────────────────────────────────────
if USE_WEBCAM:
    cap.release()
