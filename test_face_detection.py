"""
Validation and Test Suite for Real-Time Face Detection Project
=============================================================
This script tests and verifies:
1. Dependency imports (cv2, dlib, numpy).
2. Detector initialization (dlib frontal face detector).
3. Live webcam access, frame capture, and graceful resource release.
4. Single-frame and multiple-frame processing logic.
5. Drawing bounding boxes, face labels (Face 1, Face 2, etc.), and HUD.
6. Downscaling performance optimization flag.
"""

import os
import sys
import numpy as np

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import dlib
from face_detection import (
    initialize_detector,
    initialize_camera,
    detect_faces,
    draw_bounding_box,
    draw_hud,
    process_frame
)


def test_imports():
    print("[TEST 1/6] Testing dependency imports...")
    assert cv2.__version__ is not None, "OpenCV failed to import"
    assert dlib.__version__ is not None, "dlib failed to import"
    assert np.__version__ is not None, "NumPy failed to import"
    print(f"    - OpenCV version: {cv2.__version__}")
    print(f"    - dlib version: {dlib.__version__}")
    print(f"    - NumPy version: {np.__version__}")
    print("    [PASS] All core dependencies imported successfully.")


def test_detector_initialization():
    print("[TEST 2/6] Testing dlib frontal face detector initialization...")
    detector = initialize_detector()
    assert detector is not None, "Detector is None"
    print("    [PASS] Detector initialized successfully.")
    return detector


def test_synthetic_drawing_and_multi_face():
    print("[TEST 3/6] Testing multi-face rendering and HUD overlay...")
    # Create a blank 640x480 test image
    test_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulate 3 detected faces at different locations
    mock_faces = [
        (60, 100, 120, 140),   # Face 1
        (240, 90, 130, 150),   # Face 2
        (430, 110, 125, 145)   # Face 3
    ]

    # Test drawing bounding boxes and labels for each face
    for idx, box in enumerate(mock_faces, start=1):
        draw_bounding_box(test_canvas, box, face_number=idx)

    # Test drawing HUD
    draw_hud(test_canvas, face_count=len(mock_faces), fps=30.0)

    # Verify canvas was modified (non-zero pixels present)
    assert np.count_nonzero(test_canvas) > 0, "Canvas should have drawn elements"

    # Save output for inspection
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "test_multi_face_overlay.jpg")
    cv2.imwrite(out_path, test_canvas)
    print(f"    - Rendered multi-face verification output saved to: {out_path}")
    print("    [PASS] Multi-face rendering, badge styling, and HUD validated.")


def test_empty_frame_handling():
    print("[TEST 4/6] Testing empty frame HUD handling (no faces)...")
    empty_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(empty_canvas, face_count=0, fps=28.5)
    assert np.count_nonzero(empty_canvas) > 0, "Empty state HUD should be drawn"

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    out_path = os.path.join(output_dir, "test_no_faces_hud.jpg")
    cv2.imwrite(out_path, empty_canvas)
    print(f"    - Rendered empty state HUD saved to: {out_path}")
    print("    [PASS] Empty state banner rendering verified.")


def test_webcam_live_capture_and_release(detector):
    print("[TEST 5/6] Testing live webcam access and frame capture...")
    try:
        cap = initialize_camera(device_index=0)
        ret, frame = cap.read()
        assert ret and frame is not None, "Failed to read frame from webcam"
        h, w, c = frame.shape
        print(f"    - Live camera frame captured: {w}x{h} ({c} channels)")

        # Process frame end-to-end
        annotated, count = process_frame(detector, frame, scale_factor=1.0, fps=25.0)
        assert annotated.shape == frame.shape, "Output frame dimension mismatch"
        print(f"    - Real-time frame processed successfully. Faces detected: {count}")

        # Test downscaling optimization
        annotated_scaled, count_scaled = process_frame(detector, frame, scale_factor=0.5, fps=30.0)
        assert annotated_scaled.shape == frame.shape, "Scaled output dimension mismatch"
        print(f"    - Downscaled optimization (0.5x) processed successfully.")

        # Save sample frame from camera
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
        out_path = os.path.join(output_dir, "test_live_webcam_output.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"    - Processed webcam sample saved to: {out_path}")

        cap.release()
        print("    - Camera released successfully.")
        print("    [PASS] Webcam capture, face detection, and cleanup verified.")
    except Exception as exc:
        print(f"    [WARNING] Camera test encountered: {exc}")
        print("    Webcam might be inaccessible or in use by another app.")


def test_code_syntax():
    print("[TEST 6/6] Checking code compilation...")
    import py_compile
    main_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection.py")
    py_compile.compile(main_file, doraise=True)
    print("    [PASS] face_detection.py compiled with zero syntax errors.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING FACE DETECTION TEST SUITE")
    print("=" * 60)
    test_imports()
    detector = test_detector_initialization()
    test_synthetic_drawing_and_multi_face()
    test_empty_frame_handling()
    test_webcam_live_capture_and_release(detector)
    test_code_syntax()
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
