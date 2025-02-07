import math
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def process_video(input_video, output_video, model_path, confidence_threshold=0.5):
    # Load YOLO model
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

    # Set center bottom point as the reference
    center_point = (width // 2, height)

    # Define scaling and colors
    pixel_per_meter = 50
    txt_color, txt_background, bbox_color = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed or frame is empty.")
            break

        # Annotator for drawing on frames
        annotator = Annotator(frame, line_width=2)

        # Object detection and tracking
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xyxy.cpu()
        confidences = results[0].boxes.conf.cpu().tolist() if results[0].boxes.conf is not None else []
        classes = results[0].boxes.cls.cpu().tolist() if results[0].boxes.cls is not None else []
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

        for box, conf, track_id, class_id in zip(boxes, confidences, track_ids, classes):
            if conf < confidence_threshold:
                continue  # Skip objects with low confidence

            # Get class name
            class_name = model.names[int(class_id)] if class_id is not None else "Unknown"

            # Draw bounding box and ID
            annotator.box_label(box, label=f"{class_name} {track_id}", color=bbox_color)

            # Calculate the centroid of the bounding box
            centroid_x = int((box[0] + box[2]) // 2)
            centroid_y = int((box[1] + box[3]) // 2)

            # Calculate distance from center bottom
            distance = math.sqrt((centroid_x - center_point[0]) ** 2 + (centroid_y - center_point[1]) ** 2) / pixel_per_meter

            # Draw tracker line from center to the object
            cv2.line(frame, center_point, (centroid_x, centroid_y), bbox_color, 2)

            # Display distance and class name on frame
            text = f"{class_name} | Distance: {distance:.2f} m"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (centroid_x, centroid_y - text_size[1] - 10), (centroid_x + text_size[0] + 10, centroid_y), txt_background, -1)
            cv2.putText(frame, text, (centroid_x, centroid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)

        # Write frame to output video
        out.write(frame)
        cv2.imshow("Distance Calculation", frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = 0  # "person.mp4"
    output_video_path = "output_distance_calculation.avi"
    yolo_model_path = "yolo11n.pt"

    process_video(input_video_path, output_video_path, yolo_model_path, confidence_threshold=0.5)
