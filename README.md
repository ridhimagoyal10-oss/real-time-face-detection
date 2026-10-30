# Real-Time Face Detection & Web Application (OpenCV + dlib)

A high-performance, dual-interface computer vision application featuring both a **Modern Web Dashboard** and a **Desktop OpenCV GUI**. Captures video, detects human faces in real time using **dlib's Frontal Face Detector (HOG + Linear SVM)**, highlights faces with bounding boxes, assigns dynamic numbers (`Face 1`, `Face 2`, etc.), displays live face counts, and supports photo uploads.

**Author / GitHub:** [@ridhimagoyal10-oss](https://github.com/ridhimagoyal10-oss)

---

## 🌟 Key Features

1. **Dual Interfaces:**
   - **Modern Web Dashboard (`app.py`):** Browser-based UI with interactive Start/Stop camera controls, photo drag-and-drop upload, live metrics cards, and MJPEG video streaming.
   - **Desktop Native Window (`face_detection.py`):** Direct OpenCV GUI stream with keyboard shortcuts (`Q` to quit).
2. **dlib Frontal Face Detector:** Robust pre-trained HOG + Linear SVM detection architecture.
3. **In-Frame Face Numbering:** Automatically identifies and numbers each face in view (`Face 1`, `Face 2`, ...) ordered left-to-right.
4. **Photo Upload Analysis:** Upload any JPG/PNG image to detect faces, render bounding boxes, and display detailed face counts.
5. **Real-Time HUD & Telemetry:** Displays live face counter and measured FPS.
6. **Cloud Deployment Ready:** Includes `Dockerfile`, `Procfile`, and `render.yaml` for instant deployment to Render, Railway, Hugging Face Spaces, or any Docker host.

---

## 📁 Project Structure

```
face_detection_project/
│
├── app.py                     # Flask web server & REST API (Live stream, upload, controls)
├── face_detection.py          # Native desktop OpenCV application
├── requirements.txt           # Python dependencies (with Windows precompiled dlib-bin)
├── Dockerfile                 # Container image for cloud deployment (Linux / Render)
├── Procfile                   # Cloud process manager configuration
├── render.yaml                # 1-click deployment blueprint for Render
├── README.md                  # Comprehensive documentation & guides
├── .gitignore                 # Standard Python git ignore rules
├── templates/
│   └── index.html             # Modern responsive web dashboard
└── static/
    ├── css/style.css          # Dark neon cybersecurity-styled theme
    └── js/app.js              # Frontend controller & camera stream manager
```

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.8 to 3.13 (64-bit recommended).
- Ensure Python and pip are added to your system PATH.

### 2. Clone or Navigate to the Project
```bash
git clone https://github.com/ridhimagoyal10-oss/face-detection-opencv-dlib.git
cd face-detection-opencv-dlib
```

### 3. Create & Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
*(On Windows, `requirements.txt` automatically installs `dlib-bin`, avoiding the need for Visual Studio C++ build tools).*

---

## 🖥️ Running the Application

### Option A: Modern Web Dashboard (Recommended)
Run the web application server:
```bash
python app.py
```
Then open **[http://localhost:5000](http://localhost:5000)** in your browser.

- Click **▶ Start Camera** to begin live real-time detection.
- Click **⏹ Turn Off Camera** anytime to release the webcam hardware.
- Switch to the **📁 Upload Photo** tab to upload and test pictures.

### Option B: Native Desktop Window
Run the standalone OpenCV desktop application:
```bash
python face_detection.py
```
- Press **`Q`** or **`q`** to close the window and exit safely.
- Optional arguments:
  ```bash
  python face_detection.py --camera 0 --scale 0.75
  ```

---

## ☁️ Deployment Guide

### Method 1: Deploy on Render (Free Web Service)
1. Fork or push this repository to your GitHub account (`ridhimagoyal10-oss`).
2. Log in to [Render.com](https://render.com).
3. Click **New +** > **Web Service**, and connect your GitHub repository `face-detection-opencv-dlib`.
4. Choose **Docker** as the runtime (Render will automatically detect `Dockerfile`).
5. Click **Create Web Service**. Your app will be live with an HTTPS URL in a few minutes!

### Method 2: Deploy with Docker
```bash
# Build the Docker image
docker build -t face-detection-app .

# Run the container
docker run -p 5000:5000 face-detection-app
```
Then visit `http://localhost:5000`.

### Method 3: Deploy on Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Docker** as the Space SDK.
3. Push these repository files to the Space git remote.

---

## 📤 Upload to GitHub (`ridhimagoyal10-oss`)

To upload this project to your GitHub profile:

1. **Create the repository on GitHub:**
   Go to [https://github.com/new](https://github.com/new) and create a new repository:
   - **Repository name:** `face-detection-opencv-dlib`
   - **Visibility:** Public (or Private)
   - Do **NOT** initialize with a README (we already have one).

2. **Connect the remote and push:**
   Run the following commands in terminal:
   ```bash
   cd c:\Users\hp\Desktop\Smartsense\face_detection_project
   git remote add origin https://github.com/ridhimagoyal10-oss/face-detection-opencv-dlib.git
   git branch -M main
   git push -u origin main
   ```

---

## 🛠️ REST API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Main Web Dashboard UI |
| `/video_feed` | `GET` | MJPEG video stream with face overlays |
| `/api/camera/start` | `POST` | Powers on webcam stream |
| `/api/camera/stop` | `POST` | Turns off webcam and releases camera hardware |
| `/api/stats` | `GET` | Returns live face count, FPS, and camera state |
| `/api/detect_image` | `POST` | Upload an image file to receive bounding boxes and annotated base64 image |
| `/api/detect_frame` | `POST` | Process a single base64 webcam frame (for cloud deployments) |

---

## 📄 License
MIT License &copy; 2026 [Ridhima Goyal](https://github.com/ridhimagoyal10-oss).
