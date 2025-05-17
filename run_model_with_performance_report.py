import torch
import cv2
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Optimization label (update this if you're running ONNX, TensorRT, etc.)
optimization_technique = "Baseline (PyTorch - yolov5n.pt)"

# Load YOLOv5 Nano model
print(f"\n🔄 Loading model using {optimization_technique}...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')
print("✅ Model loaded.")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

# Variables for performance tracking
previous_objects = set()
frame_count = 0
inference_times = []

print("\n🚀 Starting real-time detection (10-second test)...\n")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Start inference timer
    t1 = time.time()

    # Perform object detection
    results = model(frame)

    # Stop inference timer
    t2 = time.time()
    inference_times.append(t2 - t1)

    # Object tracking logic
    detected_objects = set(results.pandas().xyxy[0]['name'].tolist())
    new_objects = detected_objects - previous_objects

    if new_objects:
        announcement = "I see " + ", ".join(new_objects)
        print(f"🗣️ {announcement}")
        engine.say(announcement)
        engine.runAndWait()
        previous_objects = detected_objects

    # Show output frame
    cv2.imshow('YOLOv5 Nano - Live Detection', results.render()[0])

    frame_count += 1
    elapsed = time.time() - start_time

    if elapsed >= 10:
        break

    # Optional: exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final metrics
total_time = time.time() - start_time
avg_fps = frame_count / total_time
avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

print("\n📊 Inference Speed Benchmark Report")
print("--------------------------------------------------")
print(f"Optimization Technique   : {optimization_technique}")
print(f"Frames Processed         : {frame_count}")
print(f"Total Elapsed Time       : {total_time:.2f} seconds")
print(f"Average FPS              : {avg_fps:.2f}")
print(f"Average Inference Time   : {avg_inference_time * 1000:.2f} ms/frame")
print("--------------------------------------------------")
