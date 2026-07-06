# VisionEdge Web — 自動透視校正 Web 應用程式

在 Chrome 瀏覽器中即時進行攝影機透視校正。使用列印的 A4 黑色邊框參考紙自動偵測紙張、計算單應性矩陣，並將整個攝影機畫面變換為俯視圖。

## 功能特色

- **全自動** — 每幀偵測 A4 黑色邊框，無需手動操作
- **全幀校正** — 校正整個桌面視角，不僅僅是紙張
- **自動白平衡** — 修正暖色/冷色光源的色偏
- **對比度增強** — 基於 CLAHE 的自適應對比度
- **時間平滑** — 減少角點偵測的抖動
- **物件偵測** — 使用 Gemini Vision API 偵測物件
- **文件掃描** — OCR 文字擷取 + 圖畫描述
- **視覺問答** — 選取區域向 AI 提問
- **現代深色 UI** — 針對 Chrome 瀏覽器最佳化

## 系統需求

- Chrome OS (ARM64) / 任何支援 Chrome 的作業系統
- HD WebCam（攝影機）
- Python 3.11+（伺服器端）
- Gemini API Key

## 本地開發

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env 填入你的 GEMINI_API_KEY

# 啟動伺服器
python main.py
```

伺服器啟動後，開啟 Chrome 瀏覽器造訪 `http://localhost:8000`

## 部署到 Render

1. 將此 repo 推送到 GitHub
2. 在 [Render](https://render.com) 建立新的 Web Service
3. 連結你的 GitHub repo
4. Render 會自動偵測 `render.yaml` 設定
5. 在 Render Dashboard 設定環境變數 `GEMINI_API_KEY`

或使用 Docker：
```bash
docker build -t visionedge .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key visionedge
```

## 列印 A4 參考紙

使用 `assets/a4_border.pdf`，列印在 A4 紙張上並放置在桌面。

## 運作原理

1. 瀏覽器透過 getUserMedia API 擷取攝影機畫面
2. 影像以 Base64 JPEG 傳送到後端 FastAPI 伺服器
3. 伺服器使用 OpenCV 偵測黑色邊框紙張
4. 計算透視變換矩陣校正畫面
5. 校正後的影像回傳到前端顯示
6. 物件偵測/掃描/問答透過 Gemini Vision API 處理

## 專案結構

```
VisionEdge/
├── main.py                  # FastAPI 入口點
├── paper_detector.py        # A4 黑色邊框偵測
├── perspective_corrector.py # 全幀透視校正
├── color_corrector.py       # 白平衡 + 對比度
├── geometry.py              # 角點排序、紙張比例驗證
├── vision_api.py            # Gemini Vision API 封裝
├── static/
│   ├── index.html           # 主頁面
│   ├── css/styles.css       # 深色主題樣式
│   └── js/app.js            # 前端應用邏輯
├── assets/
│   └── a4_border.pdf        # 可列印的參考紙
├── requirements.txt         # Python 依賴
├── render.yaml              # Render 部署設定
├── Dockerfile               # Docker 容器設定
├── .env.example             # 環境變數範本
└── README.md
```

## 疑難排解

- **攝影機無法開啟**: 確認已授予 Chrome 攝影機權限。確認沒有其他應用程式佔用攝影機。
- **紙張未偵測到**: 確保黑色邊框完全可見且光線充足。
- **校正畫面抖動**: 紙張可能被部分遮擋，確保四邊完全可見。
- **色彩偏差**: 切換白平衡 (WB) 按鈕。
- **API 錯誤**: 確認 GEMINI_API_KEY 已正確設定。

## 授權

MIT License
