# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.v1 import router as api_v1_router
from backend.services.model_loader import ModelLoader

from fastapi.responses import HTMLResponse
from pathlib import Path

# ======================================
# FastAPI App 생성
# ======================================
app = FastAPI(
    title="YOLO Web Inference API",
    description="FastAPI backend for YOLO image/video inference",
    version="1.0.0"
)

# ======================================
# CORS 설정
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 개발 단계에서는 모든 출처 허용
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# API 라우터 등록
# ======================================
app.include_router(api_v1_router, prefix="/api/v1")

# ======================================
# 정적 파일(StaticFiles) 제공 설정
# ======================================
app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)

# ======================================
# Startup Event — 모델을 서버 시작 시 1회 로드
# ======================================
@app.on_event("startup")
async def startup_event():
    print("🚀 [Startup] YOLO 모델 로딩 시작...")
    model = ModelLoader.load_model()

    if model is None:
        print(" [Startup] 모델 로딩 실패. 추론 API 사용 불가 상태입니다.")
    else:
        print(" [Startup] 모델 로딩 완료!")

# ======================================
# Health Check Endpoint
# ======================================
@app.get("/health")
def root():
    return {
        "success": True,
        "message": "YOLO Inference API Running",
        "error": None,
        "timestamp": None,
        "data": None
    }

# ======================================
# Develop 모드 실행
# ======================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
