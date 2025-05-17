import cv2
import time

# Load ONNX model
net = cv2.dnn.readNet("yolov5n.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Open webcam
cap = cv2.VideoCapture(0)

frame_count = 0
inference_times = []
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    t1 = time.time()
    net.setInput(blob)
    outputs = net.forward()
    t2 = time.time()

    inference_times.append(t2 - t1)
    frame_count += 1

    elapsed = time.time() - start_time
    if elapsed > 10:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Report
total_time = time.time() - start_time
fps = frame_count / total_time
avg_inference_time = sum(inference_times) / len(inference_times)

print("\n📊 Inference Speed Benchmark Report")
print("--------------------------------------------------")
print("Optimization Technique   : ONNX with OpenCV DNN")
print(f"Frames Processed         : {frame_count}")
print(f"Total Elapsed Time       : {total_time:.2f} seconds")
print(f"Average FPS              : {fps:.2f}")
print(f"Average Inference Time   : {avg_inference_time * 1000:.2f} ms/frame")
print("--------------------------------------------------")
