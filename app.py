"""
Web-Based Real-Time Face Detection Server & API
===============================================
Flask backend supporting:
  1. Local Webcam MJPEG Live Streaming (with dynamic Start/Stop controls)
  2. Browser/Client Webcam Processing (/api/detect_frame) for cloud deployments
  3. Static Image Upload Face Detection (/api/detect_image)
  4. Real-time JSON stats (/api/stats)

Usage:
    python app.py
    Open http://localhost:5000 in your browser.
"""

import os
import sys
import time
import base64
import threading
from typing import List, Tuple, Optional

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import dlib
except ImportError:
    print("[ERROR] dlib not installed. Run: pip install dlib-bin")
    sys.exit(1)

import numpy as np
from flask import Flask, Response, render_template, request, jsonify

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
DETECTION_SCALE = 1.0          # Downscale factor (use 0.5–0.75 for slower machines)
JPEG_QUALITY = 85              # MJPEG stream quality (1–100)

# ──────────────────────────────────────────────────────────────────────────────
# UI STYLING CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_DUPLEX
COLOR_BOX = (0, 230, 115)       # Emerald green
COLOR_WHITE = (255, 255, 255)
COLOR_TAG_BG = (25, 25, 25)     # Dark badge background
COLOR_ALERT = (30, 144, 255)    # Orange-amber alert
COLOR_HUD_BG = (20, 20, 20)
COLOR_HUD_BORDER = (50, 50, 50)

# ──────────────────────────────────────────────────────────────────────────────
# FLASK & GLOBAL STATE
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# Shared thread-safe state
lock = threading.Lock()
camera_active = False
latest_stats = {"face_count": 0, "fps": 0.0, "camera_active": False}
global_detector = None


def get_detector():
    """Lazily initialize and cache dlib detector."""
    global global_detector
    if global_detector is None:
        print("[*] Initializing dlib frontal face detector...")
        global_detector = dlib.get_frontal_face_detector()
        print("[+] dlib frontal face detector ready.")
    return global_detector


# ──────────────────────────────────────────────────────────────────────────────
# DETECTION & DRAWING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces(detector, frame: np.ndarray, scale_factor: float = 1.0) -> List[Tuple[int, int, int, int]]:
    """Run dlib frontal face detector and return list of (x, y, w, h) boxes."""
    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if 0.1 < scale_factor < 1.0:
        small = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    else:
        small = gray
        scale_factor = 1.0

    rects = detector(small, 0)
    inv = 1.0 / scale_factor
    fh, fw = frame.shape[:2]

    boxes = []
    for r in rects:
        l = max(0, min(int(r.left() * inv), fw - 1))
        t = max(0, min(int(r.top() * inv), fh - 1))
        ri = max(0, min(int(r.right() * inv), fw - 1))
        b = max(0, min(int(r.bottom() * inv), fh - 1))
        w, h = ri - l, b - t
        if w > 0 and h > 0:
            boxes.append((l, t, w, h))

    # Sort left-to-right for consistent numbering
    boxes.sort(key=lambda b: b[0])
    return boxes


def draw_bounding_box(frame: np.ndarray, box: Tuple[int, int, int, int], face_number: int) -> None:
    """Draw styled bounding box with corner accents and numbered badge."""
    x, y, w, h = box
    c = COLOR_BOX

    # Main rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2, cv2.LINE_AA)

    # Modern corner accents
    cl = max(10, min(25, int(w * 0.15)))
    ct = 3
    for (sx, sy), (dx, dy) in [
        ((x, y), (cl, 0)), ((x, y), (0, cl)),
        ((x + w, y), (-cl, 0)), ((x + w, y), (0, cl)),
        ((x, y + h), (cl, 0)), ((x, y + h), (0, -cl)),
        ((x + w, y + h), (-cl, 0)), ((x + w, y + h), (0, -cl)),
    ]:
        cv2.line(frame, (sx, sy), (sx + dx, sy + dy), c, ct, cv2.LINE_AA)

    # Label badge (Face 1, Face 2, ...)
    label = f"Face {face_number}"
    fs, ft = 0.55, 1
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, ft)
    px, py = 8, 6
    bw, bh = tw + px * 2, th + py * 2

    by1 = y - bh - 4
    if by1 < 50:
        by1 = y + 6
    by2 = by1 + bh
    bx1, bx2 = x, x + bw

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_TAG_BG, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), c, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (bx1 + px, by2 - py), FONT, fs, COLOR_WHITE, ft, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, face_count: int, fps: Optional[float] = None) -> None:
    """Draw semi-transparent top HUD bar with stats."""
    fh, fw = frame.shape[:2]
    hh = 44

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, hh), COLOR_HUD_BG, -1)
    cv2.line(overlay, (0, hh), (fw, hh), COLOR_HUD_BORDER, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "FACE DETECTOR (dlib + OpenCV)", (15, 27),
                FONT, 0.50, COLOR_WHITE, 1, cv2.LINE_AA)

    ct = f"Faces: {face_count}"
    cc = COLOR_BOX if face_count > 0 else COLOR_ALERT
    cv2.putText(frame, ct, (fw // 2 - 45, 27), FONT, 0.65, cc, 2, cv2.LINE_AA)

    fp = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
    (rw, _), _ = cv2.getTextSize(fp, FONT, 0.50, 1)
    cv2.putText(frame, fp, (fw - rw - 15, 27), FONT, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    if face_count == 0:
        msg = "No faces detected in view"
        fs2 = 0.55
        (mw, mh), _ = cv2.getTextSize(msg, FONT, fs2, 1)
        mx = (fw - mw) // 2
        my = fh - 20
        cv2.rectangle(frame, (mx - 12, my - mh - 8), (mx + mw + 12, my + 8), (20, 20, 20), -1)
        cv2.rectangle(frame, (mx - 12, my - mh - 8), (mx + mw + 12, my + 8), COLOR_ALERT, 1, cv2.LINE_AA)
        cv2.putText(frame, msg, (mx, my), FONT, fs2, COLOR_ALERT, 1, cv2.LINE_AA)


def make_placeholder_frame(text="Camera Off", subtext="Click 'Start Camera' to begin"):
    """Generate a clean dark placeholder frame when camera is not running."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (15, 18, 25)

    (tw, th), _ = cv2.getTextSize(text, FONT, 1.0, 2)
    cv2.putText(img, text, ((640 - tw) // 2, 230), FONT, 1.0, (150, 160, 180), 2, cv2.LINE_AA)

    (sw, sh), _ = cv2.getTextSize(subtext, FONT, 0.55, 1)
    cv2.putText(img, subtext, ((640 - sw) // 2, 270), FONT, 0.55, (90, 105, 130), 1, cv2.LINE_AA)

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# MJPEG STREAM GENERATOR
# ──────────────────────────────────────────────────────────────────────────────
def generate_frames():
    """Generator yielding MJPEG frames from the local webcam while camera_active is True."""
    global latest_stats, camera_active

    detector = get_detector()

    print(f"[*] Opening webcam (device {CAMERA_INDEX})...")
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        with lock:
            camera_active = False
            latest_stats["camera_active"] = False
        placeholder = make_placeholder_frame("Webcam Unavailable", "Check privacy settings or device index")
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n")
        return

    print("[+] Webcam opened. Streaming frames...")
    frame_count = 0
    fps = 0.0
    last_fps_time = time.time()

    try:
        while True:
            with lock:
                if not camera_active:
                    break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            now = time.time()
            frame_count += 1
            elapsed = now - last_fps_time
            if elapsed >= 0.3:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = now

            faces = detect_faces(detector, frame, DETECTION_SCALE)
            out = frame.copy()
            for idx, box in enumerate(faces, 1):
                draw_bounding_box(out, box, idx)
            draw_hud(out, len(faces), fps)

            with lock:
                latest_stats = {
                    "face_count": len(faces),
                    "fps": round(fps, 1),
                    "camera_active": True
                }

            _, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()
        print("[*] Webcam released cleanly. Camera is OFF.")
        with lock:
            camera_active = False
            latest_stats = {"face_count": 0, "fps": 0.0, "camera_active": False}


# ──────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Return the MJPEG stream response if active, or a single placeholder frame."""
    with lock:
        active = camera_active

    if not active:
        placeholder = make_placeholder_frame("Camera is OFF", "Click 'Start Camera' to turn it on")
        return Response(placeholder, mimetype="image/jpeg")

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    """Turn the webcam ON."""
    global camera_active
    with lock:
        camera_active = True
        latest_stats["camera_active"] = True
    return jsonify({"success": True, "camera_active": True})


@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    """Turn the webcam OFF."""
    global camera_active
    with lock:
        camera_active = False
        latest_stats["camera_active"] = False
        latest_stats["face_count"] = 0
        latest_stats["fps"] = 0.0
    return jsonify({"success": True, "camera_active": False})


@app.route("/api/stats")
def api_stats():
    """Return current detection stats as JSON."""
    with lock:
        return jsonify(latest_stats)


@app.route("/api/detect_image", methods=["POST"])
def detect_image():
    """
    Accepts an uploaded image file (or base64 image), runs dlib face detection,
    and returns an annotated image in base64 along with the detected face coordinates.
    """
    try:
        detector = get_detector()
        image_bytes = None

        if "file" in request.files:
            file = request.files["file"]
            image_bytes = file.read()
        elif request.is_json and "image" in request.json:
            b64_data = request.json["image"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)

        if not image_bytes:
            return jsonify({"success": False, "error": "No image data provided"}), 400

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Invalid image format"}), 400

        # Run face detection
        faces = detect_faces(detector, frame, scale_factor=DETECTION_SCALE)
        out = frame.copy()
        for idx, box in enumerate(faces, 1):
            draw_bounding_box(out, box, idx)
        draw_hud(out, len(faces), fps=None)

        # Encode annotated image to JPEG
        _, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        formatted_faces = [
            {"face_number": idx, "x": b[0], "y": b[1], "width": b[2], "height": b[3]}
            for idx, b in enumerate(faces, 1)
        ]

        return jsonify({
            "success": True,
            "face_count": len(faces),
            "faces": formatted_faces,
            "annotated_image": annotated_b64
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/detect_frame", methods=["POST"])
def detect_frame():
    """
    Receives a single base64 video frame from the client browser webcam,
    runs face detection, and returns the annotated frame and stats.
    Enables live detection even on cloud-deployed servers without a hardware webcam!
    """
    try:
        detector = get_detector()
        data = request.get_json(silent=True) or {}
        b64_data = data.get("frame", "")

        if not b64_data:
            return jsonify({"success": False, "error": "Missing frame data"}), 400

        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]

        image_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Could not decode frame"}), 400

        faces = detect_faces(detector, frame, scale_factor=DETECTION_SCALE)
        out = frame.copy()
        for idx, box in enumerate(faces, 1):
            draw_bounding_box(out, box, idx)
        draw_hud(out, len(faces), fps=data.get("fps"))

        _, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "face_count": len(faces),
            "annotated_frame": annotated_b64
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print("  SmartSense Real-Time Face Detection Web Application")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
