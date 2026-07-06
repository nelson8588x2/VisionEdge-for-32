"""
Gemini Vision API 封裝，用於 VisionEdge Web 版。
處理物件偵測、OCR 和圖片描述。
"""

import os
import cv2
import json
import time
import logging
import numpy as np

log = logging.getLogger(__name__)

_client = None
_last_call_time = 0.0
_MIN_CALL_INTERVAL = 2.0  # API 呼叫最小間隔（秒）
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 3.0

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _get_client():
    """延遲初始化 Gemini 客戶端。"""
    global _client
    if _client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 未設定")
        _client = genai.Client(api_key=api_key)
    return _client


_MAX_IMAGE_DIMENSION = 1280


def _resize_for_api(frame):
    """調整影像大小，使最長邊不超過 _MAX_IMAGE_DIMENSION。"""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= _MAX_IMAGE_DIMENSION:
        return frame
    scale = _MAX_IMAGE_DIMENSION / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _frame_to_part(frame, quality=85):
    """將 OpenCV BGR 畫面轉換為 Gemini Part（含縮放 + 壓縮）。"""
    from google.genai import types
    resized = _resize_for_api(frame)
    _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return types.Part.from_bytes(data=bytes(buffer), mime_type="image/jpeg")


def _call_with_retry(api_func):
    """帶速率限制和重試機制的 API 呼叫。"""
    global _last_call_time

    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)

    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            _last_call_time = time.time()
            return api_func()
        except Exception as e:
            last_err = e
            if "503" in str(e) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
                continue
            raise
    raise last_err


def _enhance_for_scan(frame):
    """增強影像對比度，使淡色手繪內容更加明顯。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr


def detect_objects(frame) -> list:
    """
    使用 Gemini Vision 偵測畫面中的物件。

    Args:
        frame: BGR 影像

    Returns:
        list of dicts: [{"name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
    """
    client = _get_client()
    h, w = frame.shape[:2]

    prompt = f"""Analyze this image and detect all visible objects.
For each object, provide:
- name: object name in English
- confidence: detection confidence from 0.0 to 1.0
- bbox: bounding box as [x1, y1, x2, y2] in pixel coordinates where image is {w}x{h}

Return ONLY a JSON array, no markdown, no explanation. Example:
[{{"name": "laptop", "confidence": 0.95, "bbox": [100, 50, 400, 300]}}]

If no objects are clearly visible, return an empty array: []"""

    image_part = _frame_to_part(frame)

    def _api_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
        )

    response = _call_with_retry(_api_call)

    try:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        objects = json.loads(text)
        if not isinstance(objects, list):
            return []
        validated = []
        for obj in objects:
            if all(k in obj for k in ("name", "confidence", "bbox")):
                bbox = obj["bbox"]
                if len(bbox) == 4:
                    bbox = [
                        max(0, min(int(bbox[0]), w)),
                        max(0, min(int(bbox[1]), h)),
                        max(0, min(int(bbox[2]), w)),
                        max(0, min(int(bbox[3]), h)),
                    ]
                    validated.append({
                        "name": str(obj["name"]),
                        "confidence": float(obj.get("confidence", 0.5)),
                        "bbox": bbox,
                    })
        return validated
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def scan_document(frame) -> dict:
    """
    掃描文件影像：擷取文字並描述圖畫/圖表。

    Args:
        frame: BGR 影像（最好已透視校正）

    Returns:
        dict: {"text": str, "descriptions": list[str]}
    """
    client = _get_client()

    prompt = """You are analyzing a scanned document image. Look VERY carefully.

This document may contain:
- Printed or handwritten text (possibly faint)
- Hand-drawn sketches, doodles, or illustrations
- Diagrams, symbols, arrows, or annotations

Your tasks:
1. Extract ALL visible text exactly as written.
2. For EVERY drawing, sketch, or non-text visual element, provide a BRIEF description.

Return ONLY a JSON object with this exact format, no markdown:
{"text": "extracted text here...", "descriptions": ["brief description 1", "brief description 2"]}

If there is no text, set text to "".
If there are no drawings, set descriptions to [].
Be thorough but concise."""

    enhanced = _enhance_for_scan(frame)
    image_part = _frame_to_part(enhanced, quality=85)

    def _api_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
        )

    response = _call_with_retry(_api_call)

    try:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        result = json.loads(text)
        return {
            "text": str(result.get("text", "")),
            "descriptions": list(result.get("descriptions", [])),
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"text": "", "descriptions": ["(解析 API 回應失敗)"]}


def ask_about_image(frame, question: str) -> str:
    """
    將裁切的圖片區域 + 使用者問題發送到 Gemini 取得回答。

    Args:
        frame: BGR 影像（裁切的感興趣區域）
        question: 使用者關於圖片的問題

    Returns:
        str: Gemini 的回答
    """
    from google.genai import types

    client = _get_client()

    prompt = f"""You are a helpful visual assistant. The user selected a region from a live camera feed and is asking about it.

User's question: {question}

Analyze the image carefully and answer the user's question directly.
- If it's a math problem, solve it step by step.
- If it's text in another language, translate as requested.
- If it's an object, describe what you see.
- Be concise and helpful. Answer in the same language as the question."""

    image_part = _frame_to_part(frame, quality=85)

    def _api_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

    response = _call_with_retry(_api_call)

    return response.text.strip()
