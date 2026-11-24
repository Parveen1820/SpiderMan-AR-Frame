
# =============================================================================
# pose_tracker.py
# Optimized Stage 5: Lightweight Shoulder Tracking
# Original camera resolution remains unchanged.
# =============================================================================

import cv2
import mediapipe as mp


class PoseTracker:

    def __init__(
        self,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        processing_width=480
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.processing_width = processing_width

        self.detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        print("[OK] Optimized PoseTracker initialized.")

    # -----------------------------------------------------------------------
    # Process a smaller copy of the frame
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
    # Get shoulder coordinates in ORIGINAL frame dimensions
    # -----------------------------------------------------------------------
    def get_shoulder_info(
        self,
        results,
        frame_w,
        frame_h,
        visibility_threshold=0.5
    ):

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        left = landmarks[
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        ]

        right = landmarks[
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ]

        if (
            left.visibility < visibility_threshold
            or right.visibility < visibility_threshold
        ):
            return None

        left_point = (
            int(left.x * frame_w),
            int(left.y * frame_h)
        )

        right_point = (
            int(right.x * frame_w),
            int(right.y * frame_h)
        )

        return left_point, right_point

    # -----------------------------------------------------------------------
    # Optional pose drawing
    # -----------------------------------------------------------------------
    def draw_detections(self, frame, results):

        if results.pose_landmarks:

            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame

    # -----------------------------------------------------------------------
    # Release resources
    # -----------------------------------------------------------------------
    def release(self):

        if self.detector:
            self.detector.close()
            self.detector = None

