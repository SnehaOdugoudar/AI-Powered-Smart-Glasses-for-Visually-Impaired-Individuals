import math
import cv2
import time
import threading
import pyttsx3
import queue
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Initialize text-to-speech engine and queue
engine = pyttsx3.init()
speech_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent backup

def speak(text):
    """Add text to the speech queue if it's not already there"""
    if not speech_queue.full():
        if text not in list(speech_queue.queue):
            try:
                speech_queue.put_nowait(text)
            except queue.Full:
                pass

def process_speech_queue():
    """Process all current items in the speech queue"""
    while not speech_queue.empty():
        try:
            text = speech_queue.get_nowait()
            engine.say(text)
            engine.runAndWait()
            speech_queue.task_done()
        except queue.Empty:
            break
        except Exception as e:
            print(f"Speech error: {e}")
            speech_queue.task_done()

def process_video(input_video, output_video, model_path, confidence_threshold=0.5, speak_interval=2):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

    center_point = (width // 2, height)
    pixel_per_meter = 50
    txt_color, txt_background, bbox_color = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

    last_spoken_time = 0
    last_positions = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video processing completed or frame is empty.")
                break

            if not speech_queue.empty():
                process_speech_queue()
            
            annotator = Annotator(frame, line_width=2)
            results = model.track(frame, persist=True)

            if results[0].boxes is None:
                continue

            boxes = results[0].boxes.xyxy.cpu()
            confidences = results[0].boxes.conf.cpu().tolist() if results[0].boxes.conf is not None else []
            classes = results[0].boxes.cls.cpu().tolist() if results[0].boxes.cls is not None else []
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

            detected_objects = []

            for box, conf, track_id, class_id in zip(boxes, confidences, track_ids, classes):
                if conf < confidence_threshold:
                    continue

                class_name = model.names[int(class_id)]
                centroid_x = int((box[0] + box[2]) // 2)
                centroid_y = int((box[1] + box[3]) // 2)

                # Determine object position
                if centroid_x < center_point[0] - 50:
                    position = "on your left"
                elif centroid_x > center_point[0] + 50:
                    position = "on your right"
                else:
                    position = "ahead"

                distance = math.sqrt((centroid_x - center_point[0]) ** 2 + 
                                  (centroid_y - center_point[1]) ** 2) / pixel_per_meter
                
                text = f"{class_name} | {position} | Distance: {distance:.2f} m"

                annotator.box_label(box, label=f"{class_name} {track_id}", color=bbox_color)
                cv2.line(frame, center_point, (centroid_x, centroid_y), bbox_color, 2)

                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (centroid_x, centroid_y - text_size[1] - 10),
                            (centroid_x + text_size[0] + 10, centroid_y), txt_background, -1)
                cv2.putText(frame, text, (centroid_x, centroid_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)

                detected_objects.append((class_name, position, distance, track_id))

            current_time = time.time()
            if current_time - last_spoken_time >= speak_interval:
                announcement_texts = []

                for obj in detected_objects:
                    class_name, position, distance, track_id = obj
                    if track_id in last_positions:
                        last_pos = last_positions[track_id]
                        movement = math.sqrt((last_pos[0] - centroid_x) ** 2 + 
                                          (last_pos[1] - centroid_y) ** 2)
                        if movement < 30:
                            continue

                    last_positions[track_id] = (centroid_x, centroid_y)
                    announcement_texts.append(f"{class_name} {position}, {distance:.1f} meters away.")

                if announcement_texts:
                    speech_text = " ".join(announcement_texts)
                    speak(speech_text)
                    last_spoken_time = current_time

            out.write(frame)
            cv2.imshow("Distance Calculation", frame)

            key = cv2.waitKey(1) & 0xFF
            print(f"Key Pressed: {key}")
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        while not speech_queue.empty():
            process_speech_queue()
        out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = 0  # Webcam
    output_video_path = "output_distance_calculation.avi"
    yolo_model_path = "yolov8n.pt"

    process_video(input_video_path, output_video_path, yolo_model_path, confidence_threshold=0.5)