"""
色彩校正工具：自動白平衡和對比度增強。
"""

import cv2
import numpy as np


class ColorCorrector:
    """Applies white balance and contrast enhancement to frames."""

    @staticmethod
    def auto_white_balance(frame):
        """
        Auto white balance using LAB color space.
        Shifts A and B channels so their mean is 128 (neutral).
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a = np.mean(lab[:, :, 1])
        avg_b = np.mean(lab[:, :, 2])

        l_norm = lab[:, :, 0] / 255.0
        lab[:, :, 1] -= (avg_a - 128) * l_norm * 1.1
        lab[:, :, 2] -= (avg_b - 128) * l_norm * 1.1

        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def enhance_contrast(frame):
        """
        Enhance contrast using CLAHE on the L channel of LAB space.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def process(self, frame, white_balance=True, contrast=True):
        """
        Apply selected corrections to a frame.

        Args:
            frame: BGR input frame
            white_balance: enable auto white balance
            contrast: enable contrast enhancement

        Returns:
            Corrected BGR frame
        """
        result = frame
        if white_balance:
            result = self.auto_white_balance(result)
        if contrast:
            result = self.enhance_contrast(result)
        return result
