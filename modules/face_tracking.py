
# =============================================================================
# face_tracking.py
# Optimized Stage 5: Lightweight Face Tracking
# Original camera resolution remains unchanged.
# =============================================================================

import cv2
import math
import numpy as np
import mediapipe as mp


class FaceTracker:
    """
    Optimized MediaPipe Face Mesh tracker.

    Returns:
    (center_x, center_y, width, height, angle)

    Coordinates are mapped to the original frame.
    """

    def __init__(
        self,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        processing_width=640
    ):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.processing_width = processing_width

        self.detector = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        print("[OK] Optimized FaceTracker initialized.")

    # -----------------------------------------------------------------------
    # Process smaller frame for better performance
    # -----------------------------------------------------------------------
    def process(self, frame):

        original_h, original_w = frame.shape[:2]

        if original_w > self.processing_width:

            scale = self.processing_width / original_w

            processing_h = max(
                1,
                int(original_h * scale)
            )

            small_frame = cv2.resize(
                frame,
                (self.processing_width, processing_h),
                interpolation=cv2.INTER_AREA
            )

        else:
            small_frame = frame

        rgb = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = self.detector.process(rgb)

        rgb.flags.writeable = True

        return results

    # -----------------------------------------------------------------------
    # Convert face landmarks to original frame coordinates
    # -----------------------------------------------------------------------
    def get_face_info(self, results, frame_w, frame_h):

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x_min = int(
            np.clip(min(xs) * frame_w, 0, frame_w - 1)
        )

        x_max = int(
            np.clip(max(xs) * frame_w, 0, frame_w - 1)
        )

        y_min = int(
            np.clip(min(ys) * frame_h, 0, frame_h - 1)
        )

        y_max = int(
            np.clip(max(ys) * frame_h, 0, frame_h - 1)
        )

        width = x_max - x_min
        height = y_max - y_min

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # Eye landmarks for head tilt
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        le_x = left_eye.x * frame_w
        le_y = left_eye.y * frame_h

        re_x = right_eye.x * frame_w
        re_y = right_eye.y * frame_h

        angle = math.degrees(
            math.atan2(
                re_y - le_y,
                re_x - le_x
            )
        )

        return (cx, cy, width, height, angle)

    # -----------------------------------------------------------------------
    # Optional face mesh debugging
    # -----------------------------------------------------------------------
    def draw_detections(self, frame, results):

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1,
                        circle_radius=1
                    )
                )

        return frame

    # -----------------------------------------------------------------------
    # Release resources
    # -----------------------------------------------------------------------
    def release(self):

        if self.detector:
            self.detector.close()
            self.detector = None
