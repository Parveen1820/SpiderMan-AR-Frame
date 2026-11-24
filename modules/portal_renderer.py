
# =============================================================================
# portal_renderer.py
# AR Frame | Stage 5: Spider-Man Character Transformation
# Optimized ROI Rendering | Connected Head, Neck and Shoulder Transformation
# Character image: assets/spiderman.png
# =============================================================================

import os
import math
import time

import cv2
import numpy as np


MIN_AREA_PX = 2_000
MIN_SIDE_PX = 30
MAX_ASPECT = 20.0

COL_GLOW_OUTER = (180, 40, 0)
COL_GLOW_MID = (0, 40, 220)
COL_GLOW_INNER = (80, 100, 255)
COL_GLOW_CORE = (200, 200, 255)
COL_CORNER_RING = (255, 220, 60)
COL_CORNER_DOT = (255, 255, 255)

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

SPIDERMAN_IMAGE_PATH = os.path.join(
    ASSETS_DIR,
    "spiderman.png"
)

CITYSCAPE_IMAGE_PATH = os.path.join(
    ASSETS_DIR,
    "cityscape.jpg"
)


# =============================================================================
# COORDINATE SMOOTHING
# =============================================================================

class CoordSmoother:

    def __init__(self, alpha=0.35):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._state = None

    def update(self, corners):
        corners = corners.astype(np.float32)

        if self._state is None:
            self._state = corners.copy()
        else:
            self._state = (
                self.alpha * corners
                + (1.0 - self.alpha) * self._state
            )

        return self._state.copy()

    def reset(self):
        self._state = None

    @property
    def is_initialised(self):
        return self._state is not None


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def _sort_corners_tl_tr_br_bl(pts):

    pts = np.asarray(pts, dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]

    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmin(diffs)]
    bl = pts[np.argmax(diffs)]

    return np.array(
        [tl, tr, br, bl],
        dtype=np.float32
    )


def _quad_area(pts):

    area = 0.0

    for i in range(len(pts)):
        j = (i + 1) % len(pts)

        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]

    return abs(area) / 2.0


def _quad_side_lengths(pts):

    return [
        float(
            np.linalg.norm(
                pts[(i + 1) % len(pts)] - pts[i]
            )
        )
        for i in range(len(pts))
    ]


def _validate_quad(pts, min_area=MIN_AREA_PX):

    sides = _quad_side_lengths(pts)

    if min(sides) < MIN_SIDE_PX:
        return False

    if _quad_area(pts) < min_area:
        return False

    if max(sides) / max(min(sides), 1e-6) > MAX_ASPECT:
        return False

    return True


def _convex_order(pts):

    hull = cv2.convexHull(
        np.asarray(pts, dtype=np.float32),
        clockwise=False
    ).reshape(-1, 2)

    if len(hull) != 4:
        return _sort_corners_tl_tr_br_bl(pts)

    start = int(np.argmin(hull.sum(axis=1)))

    return np.roll(
        hull,
        -start,
        axis=0
    ).astype(np.float32)


# =============================================================================
# ASSET LOADING
# =============================================================================

def _load_image(path, flags=cv2.IMREAD_COLOR):

    if path and os.path.isfile(path):
        return cv2.imread(path, flags)

    return None


def _make_fallback_cityscape(w=800, h=450):

    sky = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        t = y / max(h - 1, 1)

        sky[y, :] = (
            int(30 * (1 - t) + 80 * t),
            0,
            int(60 * (1 - t) + 120 * t)
        )

    rng = np.random.default_rng()

    for _ in range(28):

        bw = int(rng.integers(8, 24))

        bh = int(
            rng.integers(
                int(h * 0.2),
                int(h * 0.65)
            )
        )

        bx = int(
            rng.integers(
                0,
                max(1, w - bw)
            )
        )

        by = h - bh

        colour = (
            int(rng.integers(15, 45)),
            int(rng.integers(10, 35)),
            int(rng.integers(20, 55))
        )

        cv2.rectangle(
            sky,
            (bx, by),
            (bx + bw, h),
            colour,
            -1
        )

        for wy in range(by + 8, h - 5, 13):
            for wx in range(bx + 5, bx + bw - 3, 10):

                if rng.random() > 0.55:

                    cv2.rectangle(
                        sky,
                        (wx, wy),
                        (wx + 3, wy + 5),
                        (180, 190, 220),
                        -1
                    )

    for _ in range(150):

        x = int(rng.integers(0, w))
        y = int(rng.integers(h // 2, h))

        cv2.circle(
            sky,
            (x, y),
            1,
            (180, 190, 230),
            -1
        )

    try:

        os.makedirs(
            ASSETS_DIR,
            exist_ok=True
        )

        cv2.imwrite(
            CITYSCAPE_IMAGE_PATH,
            sky
        )

    except Exception:
        pass

    return sky


def _make_fallback_character(size=200):

    img = np.zeros(
        (size, size, 4),
        dtype=np.uint8
    )

    cx = cy = size // 2
    radius = size // 2 - 10

    cv2.circle(
        img,
        (cx, cy),
        radius,
        (0, 0, 200, 220),
        -1
    )

    for angle in range(0, 180, 30):

        x1 = int(
            cx + radius * np.cos(np.radians(angle))
        )

        y1 = int(
            cy + radius * np.sin(np.radians(angle))
        )

        x2 = int(
            cx - radius * np.cos(np.radians(angle))
        )

        y2 = int(
            cy - radius * np.sin(np.radians(angle))
        )

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 150, 200),
            1
        )

    cv2.ellipse(
        img,
        (cx - 20, cy - 10),
        (15, 10),
        -20,
        0,
        360,
        (255, 255, 255, 230),
        -1
    )

    cv2.ellipse(
        img,
        (cx + 20, cy - 10),
        (15, 10),
        20,
        0,
        360,
        (255, 255, 255, 230),
        -1
    )

    return img


# =============================================================================
# FRAME RENDERER
# =============================================================================

class FrameRenderer:

    def __init__(
        self,
        smoothing_alpha=0.35,
        min_area=MIN_AREA_PX,
        cityscape_path="",
        spiderman_path=""
    ):

        self.min_area = max(int(min_area), 1)

        self._smoother = CoordSmoother(
            alpha=smoothing_alpha
        )

        self._active = False

        # ---------------------------------------------------------------
        # CITYSCAPE
        # ---------------------------------------------------------------

        selected_cityscape = (
            cityscape_path
            if cityscape_path
            else CITYSCAPE_IMAGE_PATH
        )

        loaded_cityscape = _load_image(
            selected_cityscape
        )

        self._cityscape = (
            loaded_cityscape
            if loaded_cityscape is not None
            else _make_fallback_cityscape()
        )

        # ---------------------------------------------------------------
        # SPIDER-MAN PNG
        # ---------------------------------------------------------------

        selected_spiderman = (
            spiderman_path
            if spiderman_path
            else SPIDERMAN_IMAGE_PATH
        )

        raw = _load_image(
            selected_spiderman,
            cv2.IMREAD_UNCHANGED
        )

        if raw is None:

            print("\n[ERROR] Spider-Man image not found!")
            print("[EXPECTED PATH]", selected_spiderman)
            print("[INFO] Check that spiderman.png is inside assets.\n")

            self._spiderman = _make_fallback_character()

        elif raw.ndim == 2:

            self._spiderman = cv2.cvtColor(
                raw,
                cv2.COLOR_GRAY2BGRA
            )

        elif raw.shape[2] == 3:

            self._spiderman = cv2.cvtColor(
                raw,
                cv2.COLOR_BGR2BGRA
            )

        else:

            self._spiderman = raw

        if raw is not None:

            print(
                f"[OK] Spider-Man character loaded: "
                f"{raw.shape[1]} x {raw.shape[0]}"
            )

        # ---------------------------------------------------------------
        # SPIDER-MAN REFERENCE LANDMARKS
        # Original reference image: 447 x 559
        # ---------------------------------------------------------------

        self._reference_width = 447.0
        self._reference_height = 559.0

        self._source_left_shoulder = np.array(
            [75.0, 365.0],
            dtype=np.float32
        )

        self._source_right_shoulder = np.array(
            [372.0, 365.0],
            dtype=np.float32
        )

        self._source_head_center = np.array(
            [223.0, 170.0],
            dtype=np.float32
        )

        self._source_neck_pivot = np.array(
            [223.0, 265.0],
            dtype=np.float32
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def update(self, tips):

        required = (
            "left_thumb",
            "left_index",
            "right_thumb",
            "right_index"
        )

        if tips is None or any(
            key not in tips or tips[key] is None
            for key in required
        ):

            self._smoother.reset()
            self._active = False

            return None

        raw = np.array(
            [tips[key] for key in required],
            dtype=np.float32
        )

        if raw.shape != (4, 2):
            self._active = False
            return None

        if not np.all(np.isfinite(raw)):
            self._active = False
            return None

        ordered = _convex_order(
            _sort_corners_tl_tr_br_bl(raw)
        )

        if not _validate_quad(
            ordered,
            self.min_area
        ):

            self._active = False
            return None

        smoothed = self._smoother.update(ordered)

        self._active = True

        return np.clip(
            smoothed,
            0,
            1e6
        ).astype(np.float32)

    @property
    def is_active(self):
        return self._active

    # =========================================================================
    # DRAW FRAME - OPTIMIZED PORTAL ROI
    # =========================================================================

    def draw_frame(
        self,
        frame,
        corners,
        frame_w,
        frame_h,
        face_info=None,
        shoulder_info=None
    ):

        pts = self._clamp_corners(
            corners,
            frame_w,
            frame_h
        )

        x_min = int(np.min(pts[:, 0]))
        x_max = min(
            int(np.max(pts[:, 0])) + 1,
            frame_w
        )

        y_min = int(np.min(pts[:, 1]))
        y_max = min(
            int(np.max(pts[:, 1])) + 1,
            frame_h
        )

        box_w = max(1, x_max - x_min)
        box_h = max(1, y_max - y_min)

        city = cv2.resize(
            self._cityscape,
            (box_w, box_h),
            interpolation=cv2.INTER_LINEAR
        )

        if box_w > 30 and box_h > 30:

            city = cv2.GaussianBlur(
                city,
                (5, 5),
                0
            )

        roi = frame[
            y_min:y_max,
            x_min:x_max
        ]

        local_pts = pts.copy()

        local_pts[:, 0] -= x_min
        local_pts[:, 1] -= y_min

        portal_mask = np.zeros(
            (box_h, box_w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            portal_mask,
            [local_pts.astype(np.int32)],
            255
        )

        roi[portal_mask > 0] = city[
            portal_mask > 0
        ]

        # ---------------------------------------------------------------
        # SPIDER-MAN CHARACTER OVERLAY
        # ---------------------------------------------------------------

        if (
            face_info is not None
            and shoulder_info is not None
        ):

            try:

                cx, cy, bw, bh, face_angle = face_info

            except (TypeError, ValueError):

                cx, cy, bw, bh = face_info
                face_angle = 0.0

            frame = self._render_spiderman(
                frame=frame,
                corners=corners,
                cx=int(cx),
                cy=int(cy),
                bw=int(bw),
                bh=int(bh),
                angle=float(face_angle),
                shoulder_info=shoulder_info,
                frame_w=frame_w,
                frame_h=frame_h
            )

        # ---------------------------------------------------------------
        # NEON BORDER
        # ---------------------------------------------------------------

        self._draw_neon_border(
            frame,
            pts
        )

        return frame

    # =========================================================================
    # HUD REMOVED
    # =========================================================================

    def draw_hud(
        self,
        frame,
        corners,
        num_hands,
        fps,
        frame_w,
        frame_h,
        smoothing_alpha=0.35
    ):

        return frame

    # =========================================================================
    # NEON BORDER
    # =========================================================================

    def _draw_neon_border(self, frame, pts):

        pts_i32 = pts.astype(
            np.int32
        ).reshape((-1, 1, 2))

        pulse = 0.55 + 0.45 * (
            0.5 + 0.5 * math.sin(
                time.time() * 2 * math.pi
            )
        )

        def scale_colour(colour, factor):

            return tuple(
                int(c * factor)
                for c in colour
            )

        cv2.polylines(
            frame,
            [pts_i32],
            True,
            scale_colour(COL_GLOW_OUTER, pulse),
            14,
            cv2.LINE_AA
        )

        cv2.polylines(
            frame,
            [pts_i32],
            True,
            scale_colour(
                COL_GLOW_MID,
                0.8 + 0.2 * pulse
            ),
            7,
            cv2.LINE_AA
        )

        cv2.polylines(
            frame,
            [pts_i32],
            True,
            COL_GLOW_INNER,
            3,
            cv2.LINE_AA
        )

        cv2.polylines(
            frame,
            [pts_i32],
            True,
            COL_GLOW_CORE,
            1,
            cv2.LINE_AA
        )

        for point in pts:

            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                frame,
                (x, y),
                12,
                COL_CORNER_RING,
                2,
                cv2.LINE_AA
            )

            fill = scale_colour(
                COL_CORNER_RING,
                0.4 + 0.6 * pulse
            )

            cv2.circle(
                frame,
                (x, y),
                7,
                fill,
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                COL_CORNER_DOT,
                -1,
                cv2.LINE_AA
            )

    # =========================================================================
    # SMOOTH CONNECTED HEAD / NECK DEFORMATION
    # =========================================================================

    @staticmethod
    def _smoothstep(value):

        value = np.clip(
            value,
            0.0,
            1.0
        )

        return value * value * (3.0 - 2.0 * value)

    def _warp_connected_character(
        self,
        original_rgba,
        body_matrix,
        body_scale,
        target_neck_pivot,
        relative_head_angle,
        face_correction,
        frame_w,
        frame_h
    ):

        body_aligned = cv2.warpAffine(
            original_rgba,
            body_matrix,
            (frame_w, frame_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        yy, xx = np.mgrid[
            0:frame_h,
            0:frame_w
        ].astype(np.float32)

        map_x = xx.copy()
        map_y = yy.copy()

        angle = math.radians(
            float(relative_head_angle)
        )

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        pivot_x = float(target_neck_pivot[0])
        pivot_y = float(target_neck_pivot[1])

        transition_start = (
            pivot_y - 20.0 * body_scale
        )

        transition_end = (
            pivot_y + 125.0 * body_scale
        )

        transition_length = max(
            transition_end - transition_start,
            1.0
        )

        for _ in range(3):

            relative_x = map_x - pivot_x
            relative_y = map_y - pivot_y

            rotated_x = (
                cos_a * relative_x
                - sin_a * relative_y
            )

            rotated_y = (
                sin_a * relative_x
                + cos_a * relative_y
            )

            rotation_dx = rotated_x - relative_x
            rotation_dy = rotated_y - relative_y

            progress = (
                transition_end - map_y
            ) / transition_length

            weight = self._smoothstep(progress)

            displacement_x = weight * (
                rotation_dx + face_correction[0]
            )

            displacement_y = weight * (
                rotation_dy + face_correction[1]
            )

            map_x = xx - displacement_x
            map_y = yy - displacement_y

        return cv2.remap(
            body_aligned,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    # =========================================================================
    # SPIDER-MAN CHARACTER RENDERING - ROI OPTIMIZED
    # =========================================================================

    def _render_spiderman(
        self,
        frame,
        corners,
        cx,
        cy,
        bw,
        bh,
        angle=0.0,
        shoulder_info=None,
        frame_w=None,
        frame_h=None,
        scale=1.0
    ):

        if (
            self._spiderman is None
            or shoulder_info is None
        ):
            return frame

        if frame_w is None or frame_h is None:
            frame_h, frame_w = frame.shape[:2]

        try:

            # -----------------------------------------------------------
            # PORTAL BOUNDING RECTANGLE
            # -----------------------------------------------------------

            pts = self._clamp_corners(
                corners,
                frame_w,
                frame_h
            )

            x_min = int(np.min(pts[:, 0]))
            x_max = min(
                int(np.max(pts[:, 0])) + 1,
                frame_w
            )

            y_min = int(np.min(pts[:, 1]))
            y_max = min(
                int(np.max(pts[:, 1])) + 1,
                frame_h
            )

            roi_w = x_max - x_min
            roi_h = y_max - y_min

            if roi_w < 2 or roi_h < 2:
                return frame

            # -----------------------------------------------------------
            # SHOULDER POSITIONS
            # -----------------------------------------------------------

            left_shoulder, right_shoulder = shoulder_info

            shoulders = sorted(
                [left_shoulder, right_shoulder],
                key=lambda p: p[0]
            )

            target_left = np.asarray(
                shoulders[0],
                dtype=np.float32
            )

            target_right = np.asarray(
                shoulders[1],
                dtype=np.float32
            )

            target_vector = (
                target_right - target_left
            )

            target_distance = float(
                np.linalg.norm(target_vector)
            )

            if (
                not np.isfinite(target_distance)
                or target_distance < 40
            ):
                return frame

            # -----------------------------------------------------------
            # SOURCE CHARACTER REFERENCE POINTS
            # -----------------------------------------------------------

            original_h, original_w = (
                self._spiderman.shape[:2]
            )

            sx = original_w / self._reference_width
            sy = original_h / self._reference_height

            source_scale = np.array(
                [sx, sy],
                dtype=np.float32
            )

            source_left = (
                self._source_left_shoulder
                * source_scale
            )

            source_right = (
                self._source_right_shoulder
                * source_scale
            )

            source_head = (
                self._source_head_center
                * source_scale
            )

            source_pivot = (
                self._source_neck_pivot
                * source_scale
            )

            source_vector = (
                source_right - source_left
            )

            source_distance = float(
                np.linalg.norm(source_vector)
            )

            if source_distance < 1:
                return frame

            # -----------------------------------------------------------
            # BODY SCALE AND ROTATION
            # -----------------------------------------------------------

            body_scale = (
                target_distance / source_distance
            ) * float(scale)

            body_angle = math.atan2(
                float(target_vector[1]),
                float(target_vector[0])
            )

            body_cos = math.cos(body_angle)
            body_sin = math.sin(body_angle)

            ba = body_scale * body_cos
            bb = -body_scale * body_sin
            bc = body_scale * body_sin
            bd = body_scale * body_cos

            btx = float(target_left[0]) - (
                ba * source_left[0]
                + bb * source_left[1]
            )

            bty = float(target_left[1]) - (
                bc * source_left[0]
                + bd * source_left[1]
            )

            # -----------------------------------------------------------
            # LOCAL ROI TRANSFORMATION MATRIX
            # -----------------------------------------------------------

            body_matrix = np.float32([
                [ba, bb, btx - x_min],
                [bc, bd, bty - y_min]
            ])

            target_neck_pivot = np.array([
                ba * source_pivot[0]
                + bb * source_pivot[1]
                + btx - x_min,

                bc * source_pivot[0]
                + bd * source_pivot[1]
                + bty - y_min
            ], dtype=np.float32)

            body_angle_degrees = math.degrees(
                body_angle
            )

            relative_head_angle = float(
                np.clip(
                    float(angle) - body_angle_degrees,
                    -60.0,
                    60.0
                )
            )

            # -----------------------------------------------------------
            # PREDICTED HEAD POSITION
            # -----------------------------------------------------------

            predicted_head_center = np.array([
                ba * source_head[0]
                + bb * source_head[1]
                + btx - x_min,

                bc * source_head[0]
                + bd * source_head[1]
                + bty - y_min
            ], dtype=np.float32)

            head_angle_rad = math.radians(
                relative_head_angle
            )

            head_cos = math.cos(head_angle_rad)
            head_sin = math.sin(head_angle_rad)

            head_offset = (
                predicted_head_center
                - target_neck_pivot
            )

            rotated_head_offset = np.array([
                head_cos * head_offset[0]
                - head_sin * head_offset[1],

                head_sin * head_offset[0]
                + head_cos * head_offset[1]
            ], dtype=np.float32)

            predicted_rotated_head = (
                target_neck_pivot
                + rotated_head_offset
            )

            # -----------------------------------------------------------
            # FACE CORRECTION
            # -----------------------------------------------------------

            local_face_center = np.array(
                [cx - x_min, cy - y_min],
                dtype=np.float32
            )

            face_correction = (
                local_face_center
                - predicted_rotated_head
            )

            max_correction = max(
                2.0,
                target_distance * 0.12
            )

            correction_length = float(
                np.linalg.norm(face_correction)
            )

            if correction_length > max_correction:

                face_correction *= (
                    max_correction / correction_length
                )

            face_correction *= 0.30

            # -----------------------------------------------------------
            # RENDER CHARACTER INSIDE ROI ONLY
            # -----------------------------------------------------------

            warped_character = (
                self._warp_connected_character(
                    original_rgba=self._spiderman,
                    body_matrix=body_matrix,
                    body_scale=body_scale,
                    target_neck_pivot=target_neck_pivot,
                    relative_head_angle=relative_head_angle,
                    face_correction=face_correction,
                    frame_w=roi_w,
                    frame_h=roi_h
                )
            )

            # -----------------------------------------------------------
            # ROI ALPHA COMPOSITING
            # -----------------------------------------------------------

            frame_roi = frame[
                y_min:y_max,
                x_min:x_max
            ]

            result_roi = self._alpha_composite(
                frame_roi,
                warped_character
            )

            # -----------------------------------------------------------
            # CLIP CHARACTER TO PORTAL POLYGON
            # -----------------------------------------------------------

            local_pts = pts.copy()

            local_pts[:, 0] -= x_min
            local_pts[:, 1] -= y_min

            portal_mask = np.zeros(
                (roi_h, roi_w),
                dtype=np.uint8
            )

            cv2.fillPoly(
                portal_mask,
                [local_pts.astype(np.int32)],
                255
            )

            result_roi[portal_mask == 0] = (
                frame_roi[portal_mask == 0]
            )

            frame[
                y_min:y_max,
                x_min:x_max
            ] = result_roi

        except (
            TypeError,
            ValueError,
            IndexError,
            OverflowError,
            cv2.error
        ):

            return frame

        return frame

    # =========================================================================
    # ALPHA COMPOSITING
    # =========================================================================

    @staticmethod
    def _alpha_composite(background, foreground_rgba):

        fg = foreground_rgba[:, :, :3].astype(
            np.float32
        )

        alpha = (
            foreground_rgba[:, :, 3].astype(
                np.float32
            ) / 255.0
        )

        alpha = alpha[:, :, None]

        bg = background.astype(
            np.float32
        )

        out = (
            alpha * fg
            + (1.0 - alpha) * bg
        )

        return np.clip(
            out,
            0,
            255
        ).astype(np.uint8)

    # =========================================================================
    # GEOMETRY HELPERS
    # =========================================================================

    @staticmethod
    def _clamp_corners(corners, frame_w, frame_h):

        pts = corners.copy().astype(
            np.float32
        )

        pts[:, 0] = np.clip(
            pts[:, 0],
            0,
            frame_w - 1
        )

        pts[:, 1] = np.clip(
            pts[:, 1],
            0,
            frame_h - 1
        )

        return pts

    @staticmethod
    def _quad_size(pts):

        tl, tr, br, bl = pts

        width = int(max(
            np.linalg.norm(tr - tl),
            np.linalg.norm(br - bl)
        ))

        height = int(max(
            np.linalg.norm(bl - tl),
            np.linalg.norm(br - tr)
        ))

        return (
            max(width, 1),
            max(height, 1)
        )

    # =========================================================================
    # OPTIONAL PERSPECTIVE BACKGROUND
    # =========================================================================

    def _render_background(
        self,
        frame,
        corners,
        frame_w,
        frame_h
    ):

        pts = self._clamp_corners(
            corners,
            frame_w,
            frame_h
        )

        pw, ph = self._quad_size(pts)

        if pw < 10 or ph < 10:
            return frame

        source = np.array(
            [
                [0, 0],
                [pw - 1, 0],
                [pw - 1, ph - 1],
                [0, ph - 1]
            ],
            dtype=np.float32
        )

        city = cv2.resize(
            self._cityscape,
            (pw, ph)
        )

        matrix = cv2.getPerspectiveTransform(
            source,
            pts
        )

        warped = cv2.warpPerspective(
            city,
            matrix,
            (frame_w, frame_h)
        )

        mask = np.zeros(
            (frame_h, frame_w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [pts.astype(np.int32)],
            255
        )

        result = frame.copy()

        result[mask > 0] = warped[
            mask > 0
        ]

        return result
