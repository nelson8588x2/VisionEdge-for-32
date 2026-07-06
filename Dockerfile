FROM python:3.11-slim

# 安裝 OpenCV 所需的系統套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 複製依賴清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式原始碼
COPY main.py .
COPY paper_detector.py .
COPY perspective_corrector.py .
COPY color_corrector.py .
COPY geometry.py .
COPY vision_api.py .
COPY static/ ./static/

# 暴露埠號
EXPOSE 8000

# 啟動指令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
