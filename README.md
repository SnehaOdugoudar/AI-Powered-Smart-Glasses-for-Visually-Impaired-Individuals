# AI-Powered Smart Glasses for Visually Impaired Individuals

This project presents a wearable, AI-powered smart glasses system designed to assist visually impaired individuals by providing real-time obstacle detection and audio feedback. It leverages edge computing with a Raspberry Pi and a lightweight object detection model to ensure fast and private inference without the need for cloud connectivity.

## 🧠 Key Features
- Real-time object detection using a quantized YOLOv5 Nano model
- Edge inference on Raspberry Pi using TensorFlow Lite
- Live camera input from XIAO ESP32S3 with OV2640 module
- Spoken alerts via text-to-speech (pyttsx3) for detected obstacles
- Fully offline operation with <300ms latency and ~8 FPS performance
- Audio feedback system with duplicate-alert suppression

## 📦 Hardware Components
- Raspberry Pi 4B (4GB/8GB RAM)
- Seeed Studio XIAO ESP32S3 Sense (with OV2640 camera)
- Mini speaker (USB or 3.5mm)
- 2000mAh power bank
- Lightweight glasses frame

## 🛠️ Software Stack
- Python 3.x
- OpenCV
- TensorFlow Lite
- YOLOv5 Nano (ultralytics/yolov5)
- pyttsx3 (text-to-speech)
- ONNX (for model export/conversion)

## 🚀 How to Run
### 1. Setup Raspberry Pi
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-opencv python3-pip
pip install torch torchvision pyttsx3 numpy requests
```

### 2. Clone YOLOv5 and Export Model
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python export.py --weights yolov5n.pt --include onnx tflite
```
Copy yolov5n.onnx or yolov5n.tflite to your project directory.

### 3. Flash ESP32S3 for Camera Stream
Use the provided ESPCodeStream.ino to stream video over HTTP from the camera module. Set your Wi-Fi SSID and password in the sketch.

### 4. Run Detection Script
Update the STREAM_URL in detect_objects.py with your ESP32S3 IP.
```bash
python detect_objects.py
```
Run the script: python run_model.py

## 🧪 Performance
- Inference latency: ~180 ms per frame
- Total system latency: ~280–300 ms
- Frame rate: ~7.6 FPS (baseline), ~11.5 FPS (ONNX optimized)
- Audio response: ~120–250 ms delay

## 📁 Project Structure
```
├── detect_objects.py              # Main detection and audio feedback script
├── ESPCodeStream.ino             # ESP32S3 video stream firmware
├── yolov5n.onnx / yolov5n.tflite # Optimized model files
├── README.md                     # Project description
├── hardware/                     # Circuit and CAD models (optional)



