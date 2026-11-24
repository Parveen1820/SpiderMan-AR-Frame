
# =============================================================================
# main.py – Spider-Man AR Frame | Stage 5
# Performance Diagnostics | Fullscreen | 1280x720
# =============================================================================

import os
import time
import sys
import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------------------
try:
    from modules.hand_tracking import HandTracker
    from modules.portal_renderer import FrameRenderer
    from modules.face_tracking import FaceTracker
    from modules.pose_tracker import PoseTracker

    print("[OK] All modules imported.")

except ImportError as e:
    print(f"[ERROR] Cannot import modules: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEBCAM_INDEX = 0
WINDOW_TITLE = "Spider-Man AR Frame | Stage 5"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.5

DIAGNOSTIC_INTERVAL = 2.0


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
print("=" * 65)
print("  SPIDER-MAN AR FRAME | STAGE 5")
print("  PERFORMANCE DIAGNOSTICS ENABLED")
print("=" * 65)
print(f"  Python    : {sys.version.split()[0]}")
print(f"  OpenCV    : {cv2.__version__}")
print(f"  MediaPipe : {mp.__version__}")
print(f"  NumPy     : {np.__version__}")
print("=" * 65)
print("  D = Toggle skeleton | ESC / Q = Quit")
print("  Performance report appears in terminal every 2 seconds.")
print("=" * 65)


# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"[ERROR] Cannot open webcam (index {WEBCAM_INDEX}).")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n[OK] Webcam opened: {actual_w} x {actual_h}")


# ---------------------------------------------------------------------------
# Fullscreen window
# ---------------------------------------------------------------------------
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    WINDOW_TITLE,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print("[OK] Fullscreen display enabled.")


# ---------------------------------------------------------------------------
# Initialize trackers
# ---------------------------------------------------------------------------
tracker = None
face_tracker = None
pose_tracker = None
renderer = None

try:
    tracker = HandTracker(
        max_hands=2,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE
    )

    print("[OK] HandTracker initialized.")

    face_tracker = FaceTracker()
    print("[OK] FaceTracker initialized.")

    pose_tracker = PoseTracker()
    print("[OK] PoseTracker initialized.")

    renderer = FrameRenderer(
        cityscape_path=os.path.join("assets", "cityscape.jpg"),
        spiderman_path=os.path.join("assets", "spiderman.png")
    )

    print("[OK] FrameRenderer initialized.")

except Exception as e:
    print(f"[ERROR] Initialization failed: {e}")

    if tracker:
        tracker.release()

    if face_tracker:
        face_tracker.release()

    if pose_tracker:
        pose_tracker.release()

    cap.release()
    cv2.destroyAllWindows()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
show_skeleton = True

fps = 0.0
previous_frame_time = time.perf_counter()

diagnostic_start = time.perf_counter()
diagnostic_frames = 0

# Accumulated stage timings in milliseconds
timings = {
    "capture": 0.0,
    "hand": 0.0,
    "face": 0.0,
    "pose": 0.0,
    "render": 0.0,
    "display": 0.0,
    "total": 0.0
}


# ===========================================================================
# MAIN LOOP
# ===========================================================================
try:
    while True:

        frame_start = time.perf_counter()

        # ---------------------------------------------------------------
        # 1. Capture
        # ---------------------------------------------------------------
        stage_start = time.perf_counter()

        ret, frame = cap.read()

        capture_ms = (time.perf_counter() - stage_start) * 1000

        if not ret:
            print("[WARNING] Failed to grab frame. Retrying...")
            continue

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # ---------------------------------------------------------------
        # 2. FPS
        # ---------------------------------------------------------------
        now = time.perf_counter()

        instant_fps = 1.0 / max(now - previous_frame_time, 1e-9)

        fps = (
            0.9 * fps + 0.1 * instant_fps
            if fps > 0
            else instant_fps
        )

        previous_frame_time = now

        # ---------------------------------------------------------------
        # 3. Hand tracking
        # ---------------------------------------------------------------
        stage_start = time.perf_counter()

        results = tracker.process(frame)

        tips = tracker.get_fingertips(results, w, h)
        hands = tracker.get_individual_hands(results, w, h)

        if show_skeleton:
            frame = tracker.draw_skeleton(frame, results)

        frame = tracker.draw_fingertip_markers(frame, tips, hands)

        hand_ms = (time.perf_counter() - stage_start) * 1000

        # ---------------------------------------------------------------
        # 4. Face tracking
        # ---------------------------------------------------------------
        stage_start = time.perf_counter()

        face_results = face_tracker.process(frame)

        face_info = face_tracker.get_face_info(
            face_results,
            w,
            h
        )

        face_ms = (time.perf_counter() - stage_start) * 1000

        # ---------------------------------------------------------------
        # 5. Pose tracking
        # ---------------------------------------------------------------
        stage_start = time.perf_counter()

        pose_results = pose_tracker.process(frame)

        shoulder_info = pose_tracker.get_shoulder_info(
            pose_results,
            w,
            h
        )

        pose_ms = (time.perf_counter() - stage_start) * 1000

        # ---------------------------------------------------------------
        # 6. Portal and Spider-Man rendering
        # ---------------------------------------------------------------
        stage_start = time.perf_counter()

        corners = renderer.update(tips)

        if corners is not None:

            frame = renderer.draw_frame(
                frame,
                corners,
                w,
                h,
                face_info=face_info,
                shoulder_info=shoulder_info
            )

        frame = renderer.draw_hud(
            frame,
            corners,
            tracker.num_hands_detected,
            fps,
            w,
            h
        )

        render_ms = (time.perf_counter() - stage_start) * 1000

        # ---------------------------------------------------------------
        # 7. Debug indicator
        # ---------------------------------------------------------------
        skel_label = (
            "SKELETON: ON"
            if show_skeleton
            else "SKELETON: OFF"
        )

        skel_color = (
            (100, 255, 100)
            if show_skeleton
            else (100, 100, 100)
        )

        cv2.putText(
            frame,
            skel_label,
            (w - 160, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            skel_color,
            1,
            cv2.LINE_AA
        )

        # ---------------------------------------------------------------
        # 8. Display
        # ---------------------------------------------------------------
        stage_start = time.perf_counter()

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF

        display_ms = (time.perf_counter() - stage_start) * 1000

        # ---------------------------------------------------------------
        # 9. Accumulate performance timings
        # ---------------------------------------------------------------
        frame_total_ms = (time.perf_counter() - frame_start) * 1000

        timings["capture"] += capture_ms
        timings["hand"] += hand_ms
        timings["face"] += face_ms
        timings["pose"] += pose_ms
        timings["render"] += render_ms
        timings["display"] += display_ms
        timings["total"] += frame_total_ms

        diagnostic_frames += 1

        # ---------------------------------------------------------------
        # 10. Print diagnostic report every 2 seconds
        # ---------------------------------------------------------------
        diagnostic_now = time.perf_counter()

        elapsed = diagnostic_now - diagnostic_start

        if elapsed >= DIAGNOSTIC_INTERVAL:

            count = max(diagnostic_frames, 1)

            print("\n" + "=" * 52)
            print("       PERFORMANCE REPORT")
            print("=" * 52)

            print(f"Average FPS       : {fps:.1f}")
            print(f"Camera resolution : {w} x {h}")
            print("-" * 52)

            print(f"Webcam capture    : {timings['capture'] / count:7.2f} ms")
            print(f"Hand tracking     : {timings['hand'] / count:7.2f} ms")
            print(f"Face tracking     : {timings['face'] / count:7.2f} ms")
            print(f"Pose tracking     : {timings['pose'] / count:7.2f} ms")
            print(f"Rendering         : {timings['render'] / count:7.2f} ms")
            print(f"Display           : {timings['display'] / count:7.2f} ms")

            print("-" * 52)
            print(f"Total frame time  : {timings['total'] / count:7.2f} ms")
            print("=" * 52)

            for name in timings:
                timings[name] = 0.0

            diagnostic_frames = 0
            diagnostic_start = diagnostic_now

        # ---------------------------------------------------------------
        # 11. Keyboard controls
        # ---------------------------------------------------------------
        if key in (27, ord("q"), ord("Q")):
            print("[INFO] Quit requested.")
            break

        elif key in (ord("d"), ord("D")):

            show_skeleton = not show_skeleton

            state = "ON" if show_skeleton else "OFF"

            print(f"[INFO] Skeleton drawing: {state}")


except KeyboardInterrupt:
    print("\n[INFO] Keyboard interruption received.")

except Exception as e:
    print(f"\n[ERROR] Runtime error: {e}")

finally:

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    print("[INFO] Releasing resources...")

    if tracker:
        tracker.release()

    if face_tracker:
        face_tracker.release()

    if pose_tracker:
        pose_tracker.release()

    cap.release()
    cv2.destroyAllWindows()

    print("[DONE] Application exited cleanly.")
