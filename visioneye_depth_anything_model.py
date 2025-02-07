import math
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
from transformers import pipeline

# Load Depth-Anything model using Hugging Face pipeline
def load_depth_anything_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=0 if device == 'cuda' else -1)
    return model, device

# Process frame with Depth-Anything for depth estimation (with downscaling)
def estimate_depth_with_depth_anything(frame, depth_model, scale_factor=0.5):
    # Downscale frame
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))

    # Perform depth estimation
    depth = depth_model(image)["depth"]

    # Convert depth to NumPy array
    depth = np.array(depth)

    # Upscale depth to match original frame dimensions
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    return depth

def process_video(input_video, output_video, model_path, depth_model, device):
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
    txt_color, txt_background, bbox_color = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

    frame_count = 0  # Counter for frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed or frame is empty.")
            break

        frame_count += 1

        # Skip every other frame (process every 2nd frame)
        if frame_count % 2 != 0:
            continue

        # Perform depth estimation with downscaling
        depth_map = estimate_depth_with_depth_anything(frame, depth_model, scale_factor=0.5)

        # Annotator for drawing on frames
        annotator = Annotator(frame, line_width=2)

        # Object detection and tracking
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xyxy.cpu()
        classes = results[0].boxes.cls.cpu().tolist() if results[0].boxes.cls is not None else []
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

        for box, track_id, class_id in zip(boxes, track_ids, classes):
            # Get class name
            class_name = model.names[int(class_id)] if class_id is not None else "Unknown"

            # Draw bounding box and ID
            annotator.box_label(box, label=f"{class_name} {track_id}", color=bbox_color)

            # Calculate the centroid of the bounding box
            centroid_x = int((box[0] + box[2]) // 2)
            centroid_y = int((box[1] + box[3]) // 2)

            # Get depth value at the centroid (in meters)
            depth_value = depth_map[centroid_y, centroid_x]  # Use raw depth map without normalization

            # Draw tracker line from center to the object
            cv2.line(frame, center_point, (centroid_x, centroid_y), bbox_color, 2)

            # Display depth and class name on frame
            text = f"{class_name} | Depth: {depth_value:.2f} m"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (centroid_x, centroid_y - text_size[1] - 10), (centroid_x + text_size[0] + 10, centroid_y), txt_background, -1)
            cv2.putText(frame, text, (centroid_x, centroid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)

        # Write processed frame to output video
        out.write(frame)
        cv2.imshow("Depth Estimation", frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "person.mp4"
    output_video_path = "output_depth_estimation.avi"
    yolo_model_path = "yolo11n.pt"  # Use YOLO model for object detection

    # Load Depth-Anything model
    depth_model, device = load_depth_anything_model()

    process_video(input_video_path, output_video_path, yolo_model_path, depth_model, device)
