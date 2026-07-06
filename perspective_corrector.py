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

    def update(self, corners, frame_shape, **kwargs):
        """
        計算透視變換矩陣。

        根據偵測到的紙張角點，計算從原始畫面到俑視圖的單應性矩陣。
        自動判斷紙張方向：比較觀測到的寬高比與 ISO 216 標準比例的距離，
        自動選擇更接近的方向。若計算出的紙張尺寸超過畫面邊界過多，
        表示方向誤判，自動翻轉。

        Args:
            corners: np.ndarray (4, 2) - 偵測到的角點 (TL, TR, BR, BL)
            frame_shape: (height, width, channels) 輸入畫面的形狀
        """
        src = np.array(corners, dtype=np.float32).reshape(4, 2)
        frame_h, frame_w = frame_shape[:2]

        # 計算紙張邊長
        w_top = np.linalg.norm(src[1] - src[0])
        w_bottom = np.linalg.norm(src[2] - src[3])
        h_left = np.linalg.norm(src[3] - src[0])
        h_right = np.linalg.norm(src[2] - src[1])
        avg_w = (w_top + w_bottom) / 2.0
        avg_h = (h_left + h_right) / 2.0

        # --- 智慧方向判斷 ---
        # 計算觀測到的寬高比
        observed_ratio = avg_w / max(avg_h, 1e-6)

        # ISO 216 標準比例
        portrait_ratio = PAPER_RATIO           # ~0.707 (直向: 寬 < 高)
        landscape_ratio = 1.0 / PAPER_RATIO    # ~1.414 (橫向: 寬 > 高)

        # 哪個標準比例更接近觀測值？（用相對誤差）
        diff_landscape = abs(observed_ratio - landscape_ratio) / landscape_ratio
        diff_portrait = abs(observed_ratio - portrait_ratio) / portrait_ratio
        is_landscape_paper = diff_landscape < diff_portrait

        # 計算紙張尺寸（保持正確的 ISO 216 比例）
        paper_w_px, paper_h_px = self._compute_paper_size(
            is_landscape_paper, frame_w, frame_h
        )

        # --- 模糊邊界檢查 ---
        # 如果計算出的紙張需要大量裁剪才能放進畫面，代表方向可能誤判
        # 翻轉方向再試一次
        overflow_ratio = max(
            paper_w_px / (frame_w - 20),
            paper_h_px / (frame_h - 20),
        )
        if overflow_ratio > 1.15:
            is_landscape_paper = not is_landscape_paper
            paper_w_px, paper_h_px = self._compute_paper_size(
                is_landscape_paper, frame_w, frame_h
            )

        # 最終安全夸縮（保持比例）
        if paper_w_px > frame_w - 20:
            scale = (frame_w - 20) / paper_w_px
            paper_w_px = int(paper_w_px * scale)
            paper_h_px = int(paper_h_px * scale)
        if paper_h_px > frame_h - 20:
            scale = (frame_h - 20) / paper_h_px
            paper_w_px = int(paper_w_px * scale)
            paper_h_px = int(paper_h_px * scale)

        # 輸出尺寸與輸入相同
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

    @staticmethod
    def _compute_paper_size(is_landscape, frame_w, frame_h):
        """
        根據紙張方向和畫面尺寸，計算校正後的紙張像素尺寸。
        保持正確的 ISO 216 比例，並盡量填滿畫面。

        Returns:
            (paper_w_px, paper_h_px)
        """
        if is_landscape:
            # 橫向 A4: 寬/高 = 1.414
            target_ratio = 1.0 / PAPER_RATIO
            # 以畫面高度為基準（橫向紙張在 16:9 畫面中高度是限制因素）
            paper_h_px = int(frame_h * 0.88)
            paper_w_px = int(paper_h_px * target_ratio)
        else:
            # 直向 A4: 寬/高 = 0.707
            target_ratio = PAPER_RATIO
            # 以畫面高度為基準
            paper_h_px = int(frame_h * 0.90)
            paper_w_px = int(paper_h_px * target_ratio)

        return paper_w_px, paper_h_px

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
