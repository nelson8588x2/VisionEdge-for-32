"""
自動偵測 A4/A5 黑色邊框紙張，使用邊緣偵測和輪廓分析。
"""

import cv2
import numpy as np
from geometry import order_corners, is_valid_paper_ratio, smooth_corners


class PaperDetector:
    """
    Detects an A4/A5 paper with a printed black border in a camera frame.

    Pipeline:
      1. Grayscale + Gaussian blur
      2. Adaptive threshold to find dark regions (the black border)
      3. Canny edge detection
      4. Find contours, filter by area and polygon vertex count
      5. Validate A4 aspect ratio
      6. Temporal smoothing to reduce jitter
    """

    def __init__(self):
        self.smoothed_corners = None
        self.stable_corners = None  # corners used for perspective matrix
        self.corners_changed = False  # flag: did stable corners actually change?
        self.frames_since_detection = 0
        self.max_frames_without_detection = 30  # keep last result for ~1 sec at 30fps
        self.detection_count = 0
        self.stability_threshold = 20.0  # pixels: ignore movement below this

    def detect(self, frame):
        """
        Detect A4/A5 black-border paper in frame.

        Args:
            frame: BGR input frame

        Returns:
            corners: np.ndarray (4, 2) ordered TL, TR, BR, BL or None
            debug_frame: frame with detection overlay drawn
        """
        debug_frame = frame.copy()
        h, w = frame.shape[:2]
        min_area = h * w * 0.03  # 紙張至少佔畫面 3%（橫向 16:9 中紙張佔比較小）
        max_area = h * w * 0.70  # 紙張最多佔畫面 70%

        # Try detection at full resolution first, then downscaled
        best_quad = self._find_quad(frame, min_area, max_area)

        # Downscaled pass: reduces text noise inside the border
        if best_quad is None:
            scale = 0.4
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            s_min = sh * sw * 0.05
            s_max = sh * sw * 0.60
            quad_small = self._find_quad(small, s_min, s_max)
            if quad_small is not None:
                best_quad = quad_small / scale  # scale corners back

        if best_quad is not None:
            # Temporal smoothing (alpha=0.3 for strong smoothing)
            self.smoothed_corners = smooth_corners(
                best_quad, self.smoothed_corners, alpha=0.3
            )
            self.frames_since_detection = 0
            self.detection_count += 1

            # Stability deadzone: only update stable corners when
            # smoothed corners moved more than threshold
            self.corners_changed = False
            if self.stable_corners is None:
                self.stable_corners = self.smoothed_corners.copy()
                self.corners_changed = True
            else:
                max_shift = np.max(np.abs(self.smoothed_corners - self.stable_corners))
                if max_shift > self.stability_threshold:
                    self.stable_corners = self.smoothed_corners.copy()
                    self.corners_changed = True

            self._draw_overlay(debug_frame, self.stable_corners, detected=True)
            return self.stable_corners, debug_frame

        # No detection this frame
        self.frames_since_detection += 1

        if (self.smoothed_corners is not None and
                self.frames_since_detection <= self.max_frames_without_detection):
            # Use last known corners
            self._draw_overlay(debug_frame, self.smoothed_corners, detected=False)
            return self.smoothed_corners, debug_frame

        # Lost detection completely
        self.smoothed_corners = None
        return None, debug_frame

    def _find_quad(self, frame, min_area, max_area):
        """
        Find the best quadrilateral matching ISO 216 paper ratio (A4/A5) in the given frame.
        Uses multiple detection strategies to handle text-heavy documents.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        best_quad = None
        best_area = 0

        # Strategy 1: Adaptive threshold (good for clean borders)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel_small, iterations=2)
        thresh = cv2.erode(thresh, kernel_small, iterations=1)

        # Strategy 2: Canny edges
        edges = cv2.Canny(blurred, 40, 120)
        edges = cv2.dilate(edges, kernel_small, iterations=2)
        edges = cv2.erode(edges, kernel_small, iterations=1)

        # Strategy 3: Heavier morphology to bridge gaps caused by text
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        thresh_heavy = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 6
        )
        thresh_heavy = cv2.morphologyEx(thresh_heavy, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        thresh_heavy = cv2.erode(thresh_heavy, kernel_small, iterations=1)

        # Try all binary images
        for binary in [cv2.bitwise_or(thresh, edges), thresh_heavy]:
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    ordered = order_corners(approx.reshape(4, 2))
                    if is_valid_paper_ratio(ordered) and area > best_area:
                        best_area = area
                        best_quad = ordered

        return best_quad

    def _draw_overlay(self, frame, corners, detected=True):
        """Draw detection overlay on frame."""
        pts = corners.astype(np.int32).reshape(4, 2)
        color = (0, 255, 0) if detected else (0, 165, 255)

        cv2.polylines(frame, [pts], True, color, 3)

        labels = ["TL", "TR", "BR", "BL"]
        for i, (pt, label) in enumerate(zip(pts, labels)):
            cv2.circle(frame, tuple(pt), 8, color, -1)
            cv2.circle(frame, tuple(pt), 11, (255, 255, 255), 2)
            cv2.putText(
                frame, label,
                (pt[0] + 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                frame, label,
                (pt[0] + 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
            )

        status = "TRACKING" if detected else "USING LAST"
        cv2.putText(
            frame, status, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    def reset(self):
        """Reset detector state."""
        self.smoothed_corners = None
        self.stable_corners = None
        self.corners_changed = False
        self.frames_since_detection = 0
        self.detection_count = 0
