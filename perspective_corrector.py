"""
使用偵測到的 A4/A5 紙張角點進行透視校正。
將整個畫面變換為俯視圖 (bird's eye view)。
"""

import cv2
import numpy as np
from geometry import PAPER_RATIO


class PerspectiveCorrector:
    """
    Computes and applies a perspective transform to correct the entire
    camera frame using detected A4/A5 paper corners as reference.
    """

    def __init__(self):
        self.matrix = None
        self.inv_matrix = None
        self.output_size = None
        self.paper_bounds = None  # (x, y, w, h) of paper in corrected frame
        self.is_calibrated = False

    def update(self, corners, frame_shape):
        """
        Compute perspective transform from detected paper corners.

        The paper corners define the mapping: we know where they should be
        in a top-down view, so we compute a homography that transforms
        the entire frame accordingly.

        Args:
            corners: np.ndarray (4, 2) - detected corners (TL, TR, BR, BL)
            frame_shape: (height, width, channels) of the input frame
        """
        src = np.array(corners, dtype=np.float32).reshape(4, 2)
        frame_h, frame_w = frame_shape[:2]

        # Determine paper orientation (portrait vs landscape)
        # Average opposite edges to cancel perspective distortion
        w_top = np.linalg.norm(src[1] - src[0])
        w_bottom = np.linalg.norm(src[2] - src[3])
        h_left = np.linalg.norm(src[3] - src[0])
        h_right = np.linalg.norm(src[2] - src[1])
        avg_w = (w_top + w_bottom) / 2.0
        avg_h = (h_left + h_right) / 2.0
        is_landscape = avg_w > (avg_h * 1.5)

        if is_landscape:
            # Landscape: width is the long side
            paper_w_px = min(int(avg_w * 1.2), frame_w)
            paper_h_px = int(paper_w_px * PAPER_RATIO)
        else:
            # Portrait: height is the long side
            paper_h_px = min(int(avg_h * 1.2), frame_h)
            paper_w_px = int(paper_h_px * PAPER_RATIO)

        # Place the paper rectangle centered in the output frame
        # Output frame matches input aspect ratio
        out_w = frame_w
        out_h = frame_h

        cx = out_w / 2.0
        cy = out_h / 2.0

        dst = np.array([
            [cx - paper_w_px / 2, cy - paper_h_px / 2],
            [cx + paper_w_px / 2, cy - paper_h_px / 2],
            [cx + paper_w_px / 2, cy + paper_h_px / 2],
            [cx - paper_w_px / 2, cy + paper_h_px / 2],
        ], dtype=np.float32)

        self.matrix = cv2.getPerspectiveTransform(src, dst)
        self.inv_matrix = cv2.getPerspectiveTransform(dst, src)
        self.output_size = (out_w, out_h)
        self.paper_bounds = (
            int(cx - paper_w_px / 2),
            int(cy - paper_h_px / 2),
            int(paper_w_px),
            int(paper_h_px),
        )
        self.is_calibrated = True

    def correct(self, frame):
        """
        Apply perspective correction to the entire frame.

        Args:
            frame: BGR input frame

        Returns:
            Corrected BGR frame, or original frame if not calibrated
        """
        if not self.is_calibrated or self.matrix is None:
            return frame

        return cv2.warpPerspective(
            frame,
            self.matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(30, 30, 30),
        )

    def get_paper_only(self, corrected_frame):
        """
        Crop only the paper region from a corrected frame.

        Args:
            corrected_frame: BGR frame from correct() method

        Returns:
            Cropped BGR frame containing only the paper, or None if not calibrated
        """
        if not self.is_calibrated or self.paper_bounds is None:
            return None

        x, y, w, h = self.paper_bounds
        return corrected_frame[y:y+h, x:x+w].copy()

    def reset(self):
        """Reset calibration state."""
        self.matrix = None
        self.inv_matrix = None
        self.output_size = None
        self.paper_bounds = None
        self.is_calibrated = False
