from fastapi import APIRouter, UploadFile, File
from backend.utils.response import api_response

router = APIRouter()

@router.post("/image")
async def predict_image(file: UploadFile = File(...)):
    # 아직 추론 로직이 없으므로 placeholder 응답
    return api_response(
        success=True,
        message="이미지 추론 API 준비 완료",
        data={"filename": file.filename}
    )

@router.post("/video")
async def predict_video(file: UploadFile = File(...)):
    return api_response(
        success=True,
        message="비디오 추론 API 준비 완료",
        data={"filename": file.filename}
    )
