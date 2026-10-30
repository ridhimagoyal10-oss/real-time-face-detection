"""
Real-Time Face Detection Using Python, OpenCV, and dlib
======================================================
This application captures live video from the computer's webcam, processes
each frame to detect human faces using dlib's frontal face detector (HOG + Linear SVM),
and renders bounding boxes, dynamic face numbering (e.g., Face 1, Face 2),
a live detection count HUD, and FPS metrics.

Controls:
    Press 'Q' or 'q' to safely exit the application.
"""

import sys
import time
import argparse
from typing import List, Tuple, Optional

try:
    import cv2
except ImportError:
    print("\n[ERROR] OpenCV is not installed.")
    print("Please install OpenCV using: pip install opencv-python\n")
    sys.exit(1)

try:
    import dlib
except ImportError:
    print("\n[ERROR] dlib is not installed.")
    print("On Windows, install the prebuilt wheel with: pip install dlib-bin")
    print("On macOS/Linux, install using: pip install dlib\n")
    sys.exit(1)

import numpy as np


# ==============================================================================
# UI & STYLING CONFIGURATION
# ==============================================================================
FONT = cv2.FONT_HERSHEY_DUPLEX
COLOR_PRIMARY_GREEN = (0, 230, 115)     # Vibrant Emerald Green for bounding boxes
COLOR_TEXT_WHITE = (255, 255, 255)      # Pure White for text
COLOR_TAG_BG = (25, 25, 25)             # Dark charcoal for badge background
COLOR_ALERT_ORANGE = (30, 144, 255)     # Amber/Orange for "No faces detected"
COLOR_HEADER_BG = (20, 20, 20)          # Top HUD bar background
COLOR_HEADER_BORDER = (50, 50, 50)      # HUD divider line
COLOR_ACCENT_CYAN = (240, 200, 0)       # Cyan/Yellow accent


def initialize_camera(
    device_index: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> cv2.VideoCapture:
    """
    Initializes and returns the webcam video capture stream.

    Args:
        device_index: Index of the camera device (default is 0).
        width: Optional requested capture width in pixels.
        height: Optional requested capture height in pixels.

    Returns:
        cv2.VideoCapture: Active camera capture object.

    Raises:
        RuntimeError: If the webcam cannot be opened or accessed.
    """
    # Use cv2.CAP_DSHOW on Windows if available for faster device opening
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback to default backend if DirectShow fails
            cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to access webcam at device index {device_index}. "
            "Please ensure that your webcam is connected, enabled in OS privacy settings, "
            "and not currently in use by another application."
        )

    # Configure resolution if user supplied custom dimensions
    if width is not None and width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def initialize_detector():
    """
    Initializes and returns dlib's frontal face detector.

    Returns:
        dlib.fhog_object_detector: dlib's pre-trained HOG + Linear SVM detector.
    """
    try:
        detector = dlib.get_frontal_face_detector()
        return detector
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize dlib frontal face detector: {exc}") from exc


def detect_faces(
    detector,
    frame: np.ndarray,
    scale_factor: float = 1.0
) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces in a BGR frame using dlib's frontal face detector.

    Args:
        detector: The initialized dlib frontal face detector.
        frame: Input image/frame in BGR format (numpy array).
        scale_factor: Scale ratio for downsampling during detection (e.g. 0.75, 0.5)
                      to optimize processing speed on resource-constrained hardware.
                      1.0 preserves original resolution.

    Returns:
        List of bounding boxes represented as (x, y, w, h) in original frame coordinates.
    """
    if frame is None or frame.size == 0:
        return []

    # Convert the video frame from BGR to grayscale as required by dlib
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optional downscaling for performance optimization
    if 0.1 < scale_factor < 1.0:
        resized_gray = cv2.resize(
            gray_frame,
            (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR
        )
    else:
        resized_gray = gray_frame
        scale_factor = 1.0

    # Execute dlib frontal face detector (0 upsampling steps for real-time speed)
    dlib_rectangles = detector(resized_gray, 0)

    # Convert dlib rectangles to (x, y, w, h) format, scaling back to original frame size
    face_boxes: List[Tuple[int, int, int, int]] = []
    inv_scale = 1.0 / scale_factor

    for rect in dlib_rectangles:
        left = int(rect.left() * inv_scale)
        top = int(rect.top() * inv_scale)
        right = int(rect.right() * inv_scale)
        bottom = int(rect.bottom() * inv_scale)

        # Clip coordinates to frame boundary to prevent out-of-bounds errors
        frame_h, frame_w = frame.shape[:2]
        left = max(0, min(left, frame_w - 1))
        top = max(0, min(top, frame_h - 1))
        right = max(0, min(right, frame_w - 1))
        bottom = max(0, min(bottom, frame_h - 1))

        width = right - left
        height = bottom - top

        if width > 0 and height > 0:
            face_boxes.append((left, top, width, height))

    # Sort faces from left-to-right so numbering (Face 1, Face 2, ...) is intuitive
    face_boxes.sort(key=lambda box: box[0])

    return face_boxes


def draw_bounding_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    face_number: int
) -> None:
    """
    Draws a rectangular bounding box with corner accents and an informational
    label tag above the detected face.

    Args:
        frame: The BGR frame to modify in-place.
        box: Bounding box tuple (x, y, w, h).
        face_number: 1-indexed number assigned to this face in the current frame.
    """
    x, y, w, h = box
    color = COLOR_PRIMARY_GREEN

    # 1. Draw main rectangular bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)

    # 2. Draw aesthetic corner accents for a polished, modern look
    corner_len = max(10, min(25, int(w * 0.15)))
    corner_thickness = 3
    # Top-Left corner
    cv2.line(frame, (x, y), (x + corner_len, y), color, corner_thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + corner_len), color, corner_thickness, lineType=cv2.LINE_AA)
    # Top-Right corner
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, corner_thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, corner_thickness, lineType=cv2.LINE_AA)
    # Bottom-Left corner
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, corner_thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, corner_thickness, lineType=cv2.LINE_AA)
    # Bottom-Right corner
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thickness, lineType=cv2.LINE_AA)

    # 3. Label tag styling: Face 1, Face 2, etc.
    label_text = f"Face {face_number}"
    font_scale = 0.55
    font_thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, FONT, font_scale, font_thickness
    )

    badge_pad_x = 8
    badge_pad_y = 6
    badge_total_w = text_w + (badge_pad_x * 2)
    badge_total_h = text_h + (badge_pad_y * 2)

    # Position the badge directly above the bounding box (or just inside if close to ceiling)
    badge_y1 = y - badge_total_h - 4
    if badge_y1 < 50:  # Avoid colliding with top HUD header
        badge_y1 = y + 6
        badge_y2 = badge_y1 + badge_total_h
    else:
        badge_y2 = y - 4

    badge_x1 = x
    badge_x2 = x + badge_total_w

    # Draw solid background for high text contrast against any background
    cv2.rectangle(
        frame,
        (badge_x1, badge_y1),
        (badge_x2, badge_y2),
        COLOR_TAG_BG,
        -1
    )
    # Draw outline for the badge
    cv2.rectangle(
        frame,
        (badge_x1, badge_y1),
        (badge_x2, badge_y2),
        color,
        1,
        lineType=cv2.LINE_AA
    )

    # Render label text
    text_origin = (badge_x1 + badge_pad_x, badge_y2 - badge_pad_y)
    cv2.putText(
        frame,
        label_text,
        text_origin,
        FONT,
        font_scale,
        COLOR_TEXT_WHITE,
        font_thickness,
        lineType=cv2.LINE_AA
    )


def draw_hud(
    frame: np.ndarray,
    face_count: int,
    fps: Optional[float] = None
) -> None:
    """
    Draws a clean top HUD overlay displaying application status, total detected faces,
    current frame rate, and key shortcuts.

    Args:
        frame: The BGR frame to modify in-place.
        face_count: Total count of detected faces in the current frame.
        fps: Current measured frames per second.
    """
    frame_h, frame_w = frame.shape[:2]
    header_h = 44

    # Create a semi-transparent HUD banner across the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, header_h), COLOR_HEADER_BG, -1)
    cv2.line(overlay, (0, header_h), (frame_w, header_h), COLOR_HEADER_BORDER, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # 1. Left Title / System Name
    title_text = "FACE DETECTOR (dlib + OpenCV)"
    cv2.putText(
        frame,
        title_text,
        (15, 27),
        FONT,
        0.55,
        COLOR_TEXT_WHITE,
        1,
        lineType=cv2.LINE_AA
    )

    # 2. Center Face Counter Badge
    count_text = f"Faces: {face_count}"
    count_color = COLOR_PRIMARY_GREEN if face_count > 0 else COLOR_ALERT_ORANGE
    cv2.putText(
        frame,
        count_text,
        (frame_w // 2 - 45, 27),
        FONT,
        0.65,
        count_color,
        2,
        lineType=cv2.LINE_AA
    )

    # 3. Right Status & FPS
    fps_display = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
    exit_hint = "Quit: [Q]"
    right_info = f"{fps_display}  |  {exit_hint}"

    (r_w, _), _ = cv2.getTextSize(right_info, FONT, 0.50, 1)
    cv2.putText(
        frame,
        right_info,
        (frame_w - r_w - 15, 27),
        FONT,
        0.50,
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA
    )

    # 4. If no faces are detected, display a helpful message banner
    if face_count == 0:
        msg = "No faces detected in view"
        font_scale = 0.55
        (msg_w, msg_h), _ = cv2.getTextSize(msg, FONT, font_scale, 1)
        msg_x = (frame_w - msg_w) // 2
        msg_y = frame_h - 20

        # Draw subtle pill behind message at bottom of screen
        cv2.rectangle(
            frame,
            (msg_x - 12, msg_y - msg_h - 8),
            (msg_x + msg_w + 12, msg_y + 8),
            (20, 20, 20),
            -1
        )
        cv2.rectangle(
            frame,
            (msg_x - 12, msg_y - msg_h - 8),
            (msg_x + msg_w + 12, msg_y + 8),
            COLOR_ALERT_ORANGE,
            1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            frame,
            msg,
            (msg_x, msg_y),
            FONT,
            font_scale,
            COLOR_ALERT_ORANGE,
            1,
            lineType=cv2.LINE_AA
        )


def process_frame(
    detector,
    frame: np.ndarray,
    scale_factor: float = 1.0,
    fps: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Processes a single frame: detects faces, draws bounding boxes, numbers faces,
    and applies the HUD overlay.

    Args:
        detector: dlib frontal face detector instance.
        frame: Original input frame (BGR).
        scale_factor: Detection downsample factor for performance optimization.
        fps: Current FPS value.

    Returns:
        Tuple of (processed_frame, detected_face_count).
    """
    output_frame = frame.copy()
    faces = detect_faces(detector, output_frame, scale_factor=scale_factor)

    # Highlight and number each detected face in the current frame
    for idx, box in enumerate(faces, start=1):
        draw_bounding_box(output_frame, box, face_number=idx)

    # Draw HUD bar with total count and FPS
    draw_hud(output_frame, face_count=len(faces), fps=fps)

    return output_frame, len(faces)


def run_face_detection(
    camera_index: int = 0,
    scale_factor: float = 1.0,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> None:
    """
    Main execution loop for real-time webcam face detection.

    Args:
        camera_index: Camera device ID (default 0).
        scale_factor: Internal downscaling factor for detection performance.
        width: Optional requested capture width.
        height: Optional requested capture height.
    """
    print("=" * 65)
    print(" Real-Time Face Detection (OpenCV + dlib frontal detector)")
    print("=" * 65)
    print(f"[*] Initializing dlib frontal face detector...")
    detector = initialize_detector()
    print("[+] Detector initialized successfully.")

    print(f"[*] Connecting to webcam at device index {camera_index}...")
    cap = initialize_camera(device_index=camera_index, width=width, height=height)
    print("[+] Camera connected successfully.")
    print("[*] Starting real-time detection window.")
    print("    -> Press 'Q' or 'q' inside the video window to quit.\n")

    window_name = "Real-Time Face Detection (OpenCV & dlib)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Performance and FPS calculation variables
    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    fps_update_interval = 0.3  # Update FPS display every 300 ms for smooth readout
    last_fps_time = prev_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARNING] Failed to grab frame from camera. Exiting video loop.")
                break

            current_time = time.time()
            frame_count += 1
            elapsed = current_time - last_fps_time
            if elapsed >= fps_update_interval:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = current_time

            # Process frame with face detection, bounding boxes, labels, and HUD
            annotated_frame, face_count = process_frame(
                detector=detector,
                frame=frame,
                scale_factor=scale_factor,
                fps=fps
            )

            # Display the annotated frame
            cv2.imshow(window_name, annotated_frame)

            # Check for 'Q' or 'q' key press (1 ms wait)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("[*] Exit key pressed by user. Shutting down...")
                break

            # Handle user closing the window via the OS close [X] button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[*] Window closed by user. Shutting down...")
                break

    except KeyboardInterrupt:
        print("\n[*] Application interrupted via keyboard (Ctrl+C). Shutting down...")
    except Exception as exc:
        print(f"\n[ERROR] An unexpected error occurred: {exc}")
        raise
    finally:
        # Guarantee camera resource release and window destruction
        print("[*] Releasing camera resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Camera released and OpenCV windows closed. Goodbye!")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Face Detection using Python, OpenCV, and dlib."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Detection downscale factor (0.2 to 1.0, default: 1.0). Use e.g. 0.75 or 0.5 for faster processing."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional requested camera frame width (e.g. 640, 1280)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional requested camera frame height (e.g. 480, 720)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        run_face_detection(
            camera_index=args.camera,
            scale_factor=args.scale,
            width=args.width,
            height=args.height
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
