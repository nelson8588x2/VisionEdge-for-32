"""
幾何工具函式：角點排序、ISO 216 紙張比例驗證。
"""

import numpy as np

# A4 紙張尺寸 (mm)
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A4_RATIO = A4_WIDTH_MM / A4_HEIGHT_MM  # ~0.7071

# A5 紙張尺寸 (mm)，與 A4 相同比例 (ISO 216)
A5_WIDTH_MM = 148.0
A5_HEIGHT_MM = 210.0
A5_RATIO = A5_WIDTH_MM / A5_HEIGHT_MM  # ~0.7048

# 所有支援的紙張比例皆為 1:√2
PAPER_RATIO = A4_RATIO


def order_corners(corners):
    """
    將 4 個角點排序為：左上、右上、右下、左下。

    使用兩步法（對傾斜相機 / 旋轉紙張具有魯棒性）：
      1. 依 Y 排序分成上方對和下方對
      2. 在每對中依 X 排序決定左右

    Args:
        corners: array-like, shape (4, 2)

    Returns:
        np.ndarray shape (4, 2), dtype float32, 順序 TL, TR, BR, BL
    """
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    # 步驟 1: 依 Y 座標分成上方兩點和下方兩點
    idx_y = np.argsort(pts[:, 1])
    top_two = pts[idx_y[:2], :]
    bottom_two = pts[idx_y[2:], :]

    # 步驟 2: 在每對中，X 較小 = 左，X 較大 = 右
    idx_top = np.argsort(top_two[:, 0])
    ordered[0] = top_two[idx_top[0]]      # TL
    ordered[1] = top_two[idx_top[1]]      # TR

    idx_bottom = np.argsort(bottom_two[:, 0])
    ordered[3] = bottom_two[idx_bottom[0]]  # BL
    ordered[2] = bottom_two[idx_bottom[1]]  # BR

    return ordered


def quad_aspect_ratio(corners):
    """
    計算由 4 個有序角點 (TL, TR, BR, BL) 定義的四邊形的寬高比 (width / height)。
    """
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)

    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bottom = np.linalg.norm(pts[2] - pts[3])
    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])

    avg_w = (w_top + w_bottom) / 2.0
    avg_h = (h_left + h_right) / 2.0

    if avg_h < 1e-6:
        return 0.0

    return avg_w / avg_h


def is_valid_paper_ratio(corners, tolerance=0.40):
    """
    檢查四邊形是否符合 ISO 216 紙張寬高比 (A4/A5)。
    接受直向 (0.707) 和橫向 (1.414)。

    Args:
        corners: 有序角點 (TL, TR, BR, BL)
        tolerance: 允許的偏差比例

    Returns:
        bool
    """
    ratio = quad_aspect_ratio(corners)

    # 直向: width/height ~ 0.707
    if abs(ratio - PAPER_RATIO) / PAPER_RATIO < tolerance:
        return True

    # 橫向: width/height ~ 1.414
    landscape_ratio = 1.0 / PAPER_RATIO
    if abs(ratio - landscape_ratio) / landscape_ratio < tolerance:
        return True

    return False


def smooth_corners(new_corners, prev_corners, alpha=0.6):
    """
    使用指數移動平均平滑角點位置以減少抖動。

    Args:
        new_corners: 新偵測的角點 (4, 2)
        prev_corners: 前次平滑的角點 (4, 2) 或 None
        alpha: 新角點的權重 (0-1, 越高越靈敏)

    Returns:
        np.ndarray shape (4, 2), 平滑後的角點
    """
    new_pts = np.array(new_corners, dtype=np.float32).reshape(4, 2)

    if prev_corners is None:
        return new_pts

    prev_pts = np.array(prev_corners, dtype=np.float32).reshape(4, 2)
    return alpha * new_pts + (1.0 - alpha) * prev_pts
