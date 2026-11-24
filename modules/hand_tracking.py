# =============================================================================
# hand_tracking.py
# -----------------------------------------------------------------------------
# Wraps MediaPipe Hands (legacy Solutions API) to detect both hands and extract
# the four portal-corner fingertips:
#
#   Left  hand  –  Thumb Tip  (landmark 4)
#   Left  hand  –  Index Tip  (landmark 8)
#   Right hand  –  Thumb Tip  (landmark 4)
#   Right hand  –  Index Tip  (landmark 8)
#
# IMPORTANT – Mirror correction
# ==============================
# The webcam frame is flipped horizontally (mirror mode) BEFORE being passed
# to MediaPipe.  After flipping, what appears on the LEFT side of the image
# is physically your LEFT hand, and MediaPipe labels it correctly.
# No extra label-swapping is needed.
#
# Coordinate system
# =================
# MediaPipe returns normalised floats in [0, 1].  Every landmark is converted
# to actual pixel coordinates and clamped to [0, frame_w-1] / [0, frame_h-1]
# so downstream code is always in-bounds.
#
# Stale-data policy
# =================
# Coordinates are NEVER cached between frames.  If a hand disappears,
# get_fingertips() returns None for that hand immediately.  The caller
# is responsible for deciding how to handle partial detection.
# =============================================================================

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe landmark indices used in this project
# ---------------------------------------------------------------------------
THUMB_TIP = 4   # Tip of the thumb
INDEX_TIP = 8   # Tip of the index finger


# ---------------------------------------------------------------------------
# Visual style constants (colours in BGR)
# ---------------------------------------------------------------------------
# Each fingertip gets its own colour so they are easy to tell apart
COLOUR_LEFT_THUMB  = (0,   255, 100)   # green
COLOUR_LEFT_INDEX  = (255, 200,   0)   # cyan-yellow
COLOUR_RIGHT_THUMB = (0,   100, 255)   # orange-red
COLOUR_RIGHT_INDEX = (200,   0, 255)   # purple

COLOUR_LABEL_BG    = (20,  20,  20)    # near-black panel behind label text
COLOUR_SKELETON    = (180, 180, 180)   # light-grey hand skeleton

# Circle sizes
OUTER_RADIUS = 14   # outer ring (colour)
INNER_RADIUS =  7   # filled centre (white)
LABEL_FONT   = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE  = 0.45
LABEL_THICK  = 1


class HandTracker:
    """
    Detects up to two hands in a (mirror-flipped) BGR frame and exposes the
    four portal-corner fingertip positions.

    Typical usage
    -------------
        tracker = HandTracker()
        ...
        results   = tracker.process(frame)
        tips      = tracker.get_fingertips(results, w, h)
        frame     = tracker.draw_debug(frame, results, tips, w, h)
    """

    def __init__(
        self,
        max_hands:            int   = 2,
        detection_confidence: float = 0.7,
        tracking_confidence:  float = 0.5,
    ):
        """
        Parameters
        ----------
        max_hands            : How many hands MediaPipe tracks simultaneously.
                               Always keep this at 2 for the portal.
        detection_confidence : Minimum confidence for the initial palm detection.
                               Lower → catches more hands but has more false
                               positives.  0.7 is a good starting point.
        tracking_confidence  : Minimum confidence for landmark tracking between
                               frames.  Lower → smoother but less accurate.
        """
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles

        # Initialise the MediaPipe Hands pipeline
        self.hands = self.mp_hands.Hands(
            static_image_mode        = False,     # video mode (uses tracking)
            max_num_hands            = max_hands,
            min_detection_confidence = detection_confidence,
            min_tracking_confidence  = tracking_confidence,
        )

        # Keep track of the last seen hand count for the HUD
        self._last_num_hands = 0

    # =========================================================================
    # Core pipeline methods
    # =========================================================================

    def process(self, frame: np.ndarray):
        """
        Run MediaPipe Hands on a BGR frame and return the raw results object.

        The frame must already be horizontally flipped (mirror mode) before
        calling this method.  The flip is done in main.py immediately after
        cap.read(), so this method receives the correct orientation.

        Parameters
        ----------
        frame : BGR image (already mirrored)

        Returns
        -------
        MediaPipe Hands results object (contains multi_hand_landmarks and
        multi_handedness).
        """
        # MediaPipe requires RGB input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mark as non-writeable to avoid an internal copy inside MediaPipe
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        # Count hands for the HUD
        self._last_num_hands = (
            len(results.multi_hand_landmarks)
            if results.multi_hand_landmarks else 0
        )

        return results

    def get_fingertips(self, results, frame_w: int, frame_h: int):
        """
        Extract the four portal-corner fingertip positions from a results object.

        Hand labelling (after mirror flip)
        ------------------------------------
        MediaPipe labels hands as seen from the CAMERA'S point of view.
        After we flip the frame, the camera-left hand appears on the visual
        left – which is your physical LEFT hand.  MediaPipe's "Left" label
        therefore maps correctly to your left hand without any swap.

        Returns
        -------
        dict  if BOTH hands are detected:
            {
                "left_thumb"  : (x, y),   # pixel coords, clamped
                "left_index"  : (x, y),
                "right_thumb" : (x, y),
                "right_index" : (x, y),
            }
        None  if fewer than two hands are visible (no stale data is returned).
        """
        if not results.multi_hand_landmarks:
            return None

        left_lm  = None   # landmarks for the LEFT hand
        right_lm = None   # landmarks for the RIGHT hand

        # Zip landmark sets with their handedness classification
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            # classification[0].label is "Left" or "Right"
            label = handedness.classification[0].label
            if label == "Left":
                left_lm = hand_lm
            elif label == "Right":
                right_lm = hand_lm

        # Require BOTH hands before returning coordinates
        if left_lm is None or right_lm is None:
            return None

        return {
            "left_thumb" : self._lm_to_px(left_lm,  THUMB_TIP, frame_w, frame_h),
            "left_index" : self._lm_to_px(left_lm,  INDEX_TIP, frame_w, frame_h),
            "right_thumb": self._lm_to_px(right_lm, THUMB_TIP, frame_w, frame_h),
            "right_index": self._lm_to_px(right_lm, INDEX_TIP, frame_w, frame_h),
        }

    def get_individual_hands(self, results, frame_w: int, frame_h: int):
        """
        Like get_fingertips() but returns partial results when only ONE hand
        is visible.  Useful for the debug HUD.

        Returns
        -------
        dict with keys "left" and "right".  Each value is either a dict
        {"thumb": (x,y), "index": (x,y)} or None if that hand is not detected.

            {
                "left" : {"thumb": (x,y), "index": (x,y)}  or  None,
                "right": {"thumb": (x,y), "index": (x,y)}  or  None,
            }
        """
        result = {"left": None, "right": None}

        if not results.multi_hand_landmarks:
            return result

        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            label = handedness.classification[0].label.lower()   # "left" / "right"
            result[label] = {
                "thumb": self._lm_to_px(hand_lm, THUMB_TIP, frame_w, frame_h),
                "index": self._lm_to_px(hand_lm, INDEX_TIP, frame_w, frame_h),
            }

        return result

    # =========================================================================
    # Drawing helpers
    # =========================================================================

    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw the full 21-landmark hand skeleton on *frame* for debugging.
        Uses MediaPipe's built-in drawing utilities.

        Parameters
        ----------
        frame   : BGR image to draw on (modified in-place)
        results : output of process()

        Returns
        -------
        The same frame with skeleton overlaid.
        """
        if not results.multi_hand_landmarks:
            return frame

        for hand_lm in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_lm,
                self.mp_hands.HAND_CONNECTIONS,
                # Use default styles (grey dots, white connections)
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        return frame

    def draw_fingertip_markers(
        self,
        frame:    np.ndarray,
        tips:     dict | None,
        hands:    dict | None = None,
    ) -> np.ndarray:
        """
        Draw coloured circles and text labels at each detected fingertip.

        When *tips* is provided (both hands detected), all four markers are
        drawn with full labels.  When only *hands* is available (partial
        detection), draw whatever we have using dimmed colours and "[L]" /
        "[R]" prefixes so it's clear which hand is detected.

        Parameters
        ----------
        frame : BGR image (modified in-place)
        tips  : dict from get_fingertips() – or None
        hands : dict from get_individual_hands() – used when tips is None
        """
        if tips is not None:
            # --- Both hands: draw all four with full colours ---
            markers = [
                (tips["left_thumb"],  COLOUR_LEFT_THUMB,  "LEFT THUMB"),
                (tips["left_index"],  COLOUR_LEFT_INDEX,  "LEFT INDEX"),
                (tips["right_thumb"], COLOUR_RIGHT_THUMB, "RIGHT THUMB"),
                (tips["right_index"], COLOUR_RIGHT_INDEX, "RIGHT INDEX"),
            ]
            for pt, colour, label in markers:
                self._draw_tip(frame, pt, colour, label, alpha=1.0)

        elif hands is not None:
            # --- Partial detection: draw whatever is visible ---
            partial_map = {
                "left":  [
                    ("thumb", COLOUR_LEFT_THUMB,  "[L] THUMB"),
                    ("index", COLOUR_LEFT_INDEX,  "[L] INDEX"),
                ],
                "right": [
                    ("thumb", COLOUR_RIGHT_THUMB, "[R] THUMB"),
                    ("index", COLOUR_RIGHT_INDEX, "[R] INDEX"),
                ],
            }
            for side, tip_defs in partial_map.items():
                if hands[side] is None:
                    continue
                for key, colour, label in tip_defs:
                    pt = hands[side][key]
                    self._draw_tip(frame, pt, colour, label, alpha=0.6)

        return frame

    def draw_debug_hud(
        self,
        frame:  np.ndarray,
        tips:   dict | None,
        fps:    float,
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        """
        Draw the Stage-2 information panel (top-left corner) showing:
          • FPS
          • Number of hands detected
          • Whether both hands are visible
          • The pixel coordinates of all four fingertips (or "---" if missing)

        Parameters
        ----------
        frame   : BGR image (modified in-place)
        tips    : dict from get_fingertips() – or None
        fps     : current smoothed FPS value
        frame_w : frame width  (pixels)
        frame_h : frame height (pixels)
        """
        num_hands = self._last_num_hands
        both_ok   = tips is not None

        # Build the text lines for the panel
        lines = [
            (f"FPS          : {fps:5.1f}",           (200, 230, 200)),
            (f"Hands found  : {num_hands}",            (200, 230, 200)),
            (f"Both ready   : {'YES  ✓' if both_ok else 'NO   ✗'}",
             (0, 255, 120) if both_ok else (80, 80, 255)),
            ("",                                        (0, 0, 0)),   # spacer
            # --- Fingertip coordinates ---
        ]

        # Four fingertip coordinate rows
        tip_rows = [
            ("L-THUMB ", COLOUR_LEFT_THUMB,  "left_thumb"),
            ("L-INDEX ", COLOUR_LEFT_INDEX,  "left_index"),
            ("R-THUMB ", COLOUR_RIGHT_THUMB, "right_thumb"),
            ("R-INDEX ", COLOUR_RIGHT_INDEX, "right_index"),
        ]
        for label, colour, key in tip_rows:
            if both_ok and tips:
                x, y  = tips[key]
                value = f"({x:4d}, {y:4d})"
            else:
                value = "    ---"
            lines.append((f"{label}: {value}", colour))

        lines.append(("",               (0, 0, 0)))          # spacer
        lines.append(("ESC / Q  : quit", (160, 160, 160)))
        lines.append(("Stage 3  : AR Frame", (0, 200, 255)))

        # --- Draw semi-transparent panel background ---
        line_h   = 24
        pad      = 10
        panel_w  = 320
        panel_h  = len(lines) * line_h + pad * 2
        x0, y0   = 8, 8
        x1, y1   = x0 + panel_w, y0 + panel_h

        # Clamp to frame
        x1 = min(x1, frame_w - 1)
        y1 = min(y1, frame_h - 1)

        roi = frame[y0:y1, x0:x1]
        dark_panel = np.full_like(roi, (15, 15, 15))
        cv2.addWeighted(dark_panel, 0.72, roi, 0.28, 0, roi)
        frame[y0:y1, x0:x1] = roi

        # Border
        cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1)

        # --- Write each text line ---
        for i, (text, colour) in enumerate(lines):
            if not text:
                continue
            ty = y0 + pad + i * line_h + 12
            cv2.putText(
                frame, text,
                (x0 + pad, ty),
                LABEL_FONT, LABEL_SCALE,
                colour, LABEL_THICK, cv2.LINE_AA,
            )

        # --- Bottom banner ---
        banner_h = 36
        cv2.rectangle(frame, (0, frame_h - banner_h), (frame_w, frame_h), (0, 0, 160), -1)
        cv2.putText(
            frame,
            "  SPIDER-MAN AR FRAME  |  Stage 2 – Hand Tracking  ",
            (10, frame_h - 10),
            cv2.FONT_HERSHEY_DUPLEX, 0.68,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        return frame

    # =========================================================================
    # Housekeeping
    # =========================================================================

    @property
    def num_hands_detected(self) -> int:
        """Number of hands detected in the most recent frame."""
        return self._last_num_hands

    def release(self):
        """Release MediaPipe resources.  Call when the application exits."""
        self.hands.close()

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _lm_to_px(landmarks, idx: int, frame_w: int, frame_h: int):
        """
        Convert a normalised MediaPipe landmark to clamped pixel coordinates.

        MediaPipe normalises x and y to [0.0, 1.0] relative to the image size.
        Multiply by the actual dimensions and clamp to avoid out-of-bounds.
        """
        lm = landmarks.landmark[idx]
        x  = int(np.clip(lm.x * frame_w, 0, frame_w - 1))
        y  = int(np.clip(lm.y * frame_h, 0, frame_h - 1))
        return (x, y)

    @staticmethod
    def _draw_tip(
        frame:  np.ndarray,
        pt:     tuple,
        colour: tuple,
        label:  str,
        alpha:  float = 1.0,
    ):
        """
        Draw an annotated fingertip marker at pixel position *pt*.

        The marker consists of:
          • An outer coloured ring  (radius = OUTER_RADIUS)
          • A white filled centre   (radius = INNER_RADIUS)
          • A text label with a dark background box above the circle

        *alpha* dims the colour when a hand is only partially detected.
        """
        x, y = pt

        # Dim colour for partial detections
        if alpha < 1.0:
            colour = tuple(int(c * alpha) for c in colour)

        # Outer ring
        cv2.circle(frame, (x, y), OUTER_RADIUS, colour, 2, cv2.LINE_AA)
        # Inner filled dot
        cv2.circle(frame, (x, y), INNER_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
        # Coloured centre dot
        cv2.circle(frame, (x, y), INNER_RADIUS - 3, colour, -1, cv2.LINE_AA)

        # --- Label with background box ---
        (tw, th), baseline = cv2.getTextSize(
            label, LABEL_FONT, LABEL_SCALE, LABEL_THICK
        )
        lx = x - tw // 2        # centre label over the circle
        ly = y - OUTER_RADIUS - 6   # place it just above the circle

        # Background box
        cv2.rectangle(
            frame,
            (lx - 3, ly - th - 2),
            (lx + tw + 3, ly + baseline),
            COLOUR_LABEL_BG, -1,
        )
        # Text
        cv2.putText(
            frame, label,
            (lx, ly),
            LABEL_FONT, LABEL_SCALE,
            colour, LABEL_THICK, cv2.LINE_AA,
        )
