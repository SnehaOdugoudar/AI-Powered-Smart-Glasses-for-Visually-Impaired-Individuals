import math
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
import depth_pro

# Load ML Depth Pro model
def load_ml_depth_pro_model():
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    return model, transform

# Process frame with ML Depth Pro for depth estimation
def estimate_depth_with_ml_depth_pro(frame, depth_model, transform):
    # Prepare the input image for ML Depth Pro
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image)  # Apply the transform provided by depth_pro

    # Create a placeholder for focal length in pixels (adjust as needed)
    f_px = torch.tensor([1000.0])  # Example focal length as a tensor; replace with actual value if known

    # Predict depth
    prediction = depth_model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in meters
    return depth

def process_video(input_video, output_video, model_path, depth_model, transform):
    # Load YOLO model
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video)

    # Reduce video resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

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

    frame_count = 0
    batch_size = 5
    batch_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed or frame is empty.")
            break

        frame_count += 1

        # Reduce frame rate by skipping frames
        if frame_count % 2 != 0:  # Process every 2nd frame
            continue

        # Downscale depth estimation
        small_frame = cv2.resize(frame, (320, 240))
        batch_frames.append(small_frame)

        # Process in batches
        if len(batch_frames) == batch_size:
            batch_depth_maps = []
            for bf in batch_frames:
                depth_map = estimate_depth_with_ml_depth_pro(bf, depth_model, transform)
                depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))  # Upscale depth map
                batch_depth_maps.append(depth_map)

            for bf, depth_map in zip(batch_frames, batch_depth_maps):
                # Annotator for drawing on frames
                annotator = Annotator(bf, line_width=2)

                # Object detection and tracking
                results = model.track(bf, persist=True)
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

                    # Get depth value at the centroid
                    depth_value = depth_map[centroid_y, centroid_x]

                    # Draw tracker line from center to the object
                    cv2.line(bf, center_point, (centroid_x, centroid_y), bbox_color, 2)

                    # Display depth and class name on frame
                    text = f"{class_name} | Depth: {depth_value:.2f} m"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(bf, (centroid_x, centroid_y - text_size[1] - 10), (centroid_x + text_size[0] + 10, centroid_y), txt_background, -1)
                    cv2.putText(bf, text, (centroid_x, centroid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)

                # Write processed frame to output video
                out.write(bf)
                cv2.imshow("Depth Estimation", bf)

            batch_frames = []

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
    yolo_model_path = "yolo11s.pt"  # Use smaller YOLO model for optimization

    # Load ML Depth Pro model
    depth_model, transform = load_ml_depth_pro_model()

    process_video(input_video_path, output_video_path, yolo_model_path, depth_model, transform)
