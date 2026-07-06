"""
VisionEdge Web — 自動透視校正 Web 應用程式
FastAPI 入口點，提供靜態前端檔案和影像處理 API。
"""

import os
import base64
import logging
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

from paper_detector import PaperDetector
from perspective_corrector import PerspectiveCorrector
from color_corrector import ColorCorrector
import vision_api

# 設定日誌
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="VisionEdge Web", version="2.0.0")

# ============================================================
# 全域狀態（單一使用者 Demo 模式）
# ============================================================
detector = PaperDetector()
corrector = PerspectiveCorrector()
color_corrector_instance = ColorCorrector()


# ============================================================
# Pydantic 模型
# ============================================================
class FrameRequest(BaseModel):
    """前端傳送的影像幀（Base64 編碼的 JPEG）。"""
    image: str  # base64 編碼的 JPEG
    wb_enabled: bool = True
    contrast_enabled: bool = False


class ProcessResponse(BaseModel):
    """校正處理回應。"""
    original_display: Optional[str] = None  # base64 JPEG（含偵測覆疊）
    corrected_display: Optional[str] = None  # base64 JPEG（校正後）
    status: str = "idle"
    status_type: str = "idle"  # ok, warn, error, idle
    is_calibrated: bool = False


class DetectRequest(BaseModel):
    """物件偵測請求。"""
    image: str  # base64 JPEG
    wb_enabled: bool = True
    contrast_enabled: bool = False


class DetectResponse(BaseModel):
    """物件偵測回應。"""
    objects: list = []
    display: Optional[str] = None  # 帶有標記框的 base64 JPEG
    status: str = ""


class ScanRequest(BaseModel):
    """文件掃描請求。"""
    image: str  # base64 JPEG（校正後的畫面）


class ScanResponse(BaseModel):
    """文件掃描回應。"""
    text: str = ""
    descriptions: list = []
    status: str = ""


class ChatRequest(BaseModel):
    """視覺問答請求。"""
    image: str  # base64 JPEG（裁切區域）
    question: str


class ChatResponse(BaseModel):
    """視覺問答回應。"""
    answer: str = ""
    status: str = ""


# ============================================================
# 工具函式
# ============================================================
def decode_frame(base64_str: str) -> np.ndarray:
    """將 Base64 JPEG 解碼為 OpenCV BGR 影像。"""
    try:
        # 移除 data URL 前綴（如果有）
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("無法解碼影像")
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"影像解碼失敗: {str(e)}")


def encode_frame(frame: np.ndarray, quality: int = 85) -> str:
    """將 OpenCV BGR 影像編碼為 Base64 JPEG。"""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


def draw_detections(frame: np.ndarray, objects: list) -> np.ndarray:
    """在畫面上繪製物件偵測的標記框。"""
    result = frame.copy()
    for obj in objects:
        bbox = obj["bbox"]
        name = obj["name"]
        conf = obj["confidence"]
        x1, y1, x2, y2 = bbox

        # 依信心度上色
        if conf >= 0.8:
            color = (0, 230, 118)   # 綠色
        elif conf >= 0.5:
            color = (0, 176, 255)   # 藍色
        else:
            color = (255, 152, 0)   # 橙色

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(result, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return result


# ============================================================
# API 路由
# ============================================================
@app.post("/api/process", response_model=ProcessResponse)
async def process_frame(req: FrameRequest):
    """
    處理一幀影像：偵測紙張、透視校正。
    這是主要的即時處理端點。
    """
    frame = decode_frame(req.image)

    # 紙張偵測
    corners, debug_frame = detector.detect(frame)

    if corners is not None:
        if detector.corners_changed:
            corrector.update(corners, frame.shape)

        if detector.frames_since_detection == 0:
            status_type = "ok"
            status = "紙張偵測成功 — 已校準"
        else:
            status_type = "warn"
            status = "使用上次偵測結果"

        corrected_raw = corrector.correct(frame)
        display_corrected = color_corrector_instance.process(
            corrected_raw, req.wb_enabled, req.contrast_enabled
        )
        corrected_b64 = encode_frame(display_corrected)
    else:
        if corrector.is_calibrated:
            status_type = "warn"
            status = "紙張遺失 — 使用上次校準"
            corrected_raw = corrector.correct(frame)
            display_corrected = color_corrector_instance.process(
                corrected_raw, req.wb_enabled, req.contrast_enabled
            )
            corrected_b64 = encode_frame(display_corrected)
        else:
            status_type = "error"
            status = "未偵測到紙張 — 未校準"
            corrected_b64 = None

    # 色彩校正原始畫面（含偵測覆疊）
    display_original = color_corrector_instance.process(
        debug_frame, req.wb_enabled, req.contrast_enabled
    )
    original_b64 = encode_frame(display_original)

    return ProcessResponse(
        original_display=original_b64,
        corrected_display=corrected_b64,
        status=status,
        status_type=status_type,
        is_calibrated=corrector.is_calibrated,
    )


@app.post("/api/detect", response_model=DetectResponse)
async def detect_objects_endpoint(req: DetectRequest):
    """使用 Gemini Vision 進行物件偵測。"""
    frame = decode_frame(req.image)

    display_frame = color_corrector_instance.process(
        frame, req.wb_enabled, req.contrast_enabled
    )

    try:
        objects = vision_api.detect_objects(frame)
        display_with_boxes = draw_detections(display_frame, objects)
        status = f"偵測到 {len(objects)} 個物件" if objects else "未偵測到物件"
        return DetectResponse(
            objects=objects,
            display=encode_frame(display_with_boxes),
            status=status,
        )
    except Exception as e:
        log.error(f"偵測失敗: {e}")
        return DetectResponse(
            objects=[],
            display=encode_frame(display_frame),
            status=f"偵測錯誤: {str(e)}",
        )


@app.post("/api/scan", response_model=ScanResponse)
async def scan_document_endpoint(req: ScanRequest):
    """掃描文件：OCR + 圖畫描述。"""
    frame = decode_frame(req.image)

    # 如果已校準，使用校正後的紙張區域
    if corrector.is_calibrated:
        corrected = corrector.correct(frame)
        paper_only = corrector.get_paper_only(corrected)
        scan_frame = paper_only if paper_only is not None else corrected
    else:
        scan_frame = frame

    try:
        result = vision_api.scan_document(scan_frame)
        return ScanResponse(
            text=result.get("text", ""),
            descriptions=result.get("descriptions", []),
            status="掃描完成",
        )
    except Exception as e:
        log.error(f"掃描失敗: {e}")
        return ScanResponse(status=f"掃描錯誤: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """視覺問答：發送裁切區域 + 問題到 Gemini。"""
    frame = decode_frame(req.image)

    try:
        answer = vision_api.ask_about_image(frame, req.question)
        return ChatResponse(answer=answer, status="回答完成")
    except Exception as e:
        log.error(f"問答失敗: {e}")
        return ChatResponse(status=f"問答錯誤: {str(e)}")


@app.post("/api/reset")
async def reset_calibration():
    """重置偵測器和校正器狀態。"""
    detector.reset()
    corrector.reset()
    return {"status": "已重置"}


@app.get("/api/health")
async def health_check():
    """健康檢查端點。"""
    return {"status": "ok", "calibrated": corrector.is_calibrated}


# ============================================================
# 靜態檔案服務
# ============================================================
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
async def serve_index():
    """提供主頁面。"""
    return FileResponse(os.path.join(static_dir, "index.html"))


# 掛載靜態檔案目錄
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================
# 啟動入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

