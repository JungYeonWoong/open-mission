# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.v1 import router as api_v1_router

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# 정적 파일(StaticFiles) 제공 설정
# ======================================
app.mount(
    "/static",
    StaticFiles(directory="backend/static"),
    name="static"
)

# ======================================
# API 라우터 등록
# ======================================
app.include_router(api_v1_router, prefix="/api/v1")


# ======================================
# Health Check Endpoint
# ======================================
@app.get("/")
def root():
    return api_response(
        success=True,
        message="YOLO Inference API Running",
        data=None
    )

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
