from fastapi import APIRouter, UploadFile, File
from backend.utils.response import api_response
from backend.services.image_service import ImageService

router = APIRouter()

@router.post("/image")
async def predict_image(file: UploadFile = File(...)):
    # 파일 확장자 기본 검증
    allowed_ext = ["jpg", "jpeg", "png"]
    ext = file.filename.split(".")[-1].lower()

    if ext not in allowed_ext:
        return api_response(
            success=False,
            message="지원하지 않는 파일 형식입니다.",
            error=f"Allowed: {allowed_ext}, Received: {ext}",
            data=None
        )

    # 서비스 레이어 호출 (아직 placeholder)
    filename = await ImageService.handle_image_upload(file)

    return api_response(
        success=True,
        message="이미지 업로드 성공",
        data={"filename": filename}
    )

@router.post("/video")
async def predict_video(file: UploadFile = File(...)):
    return api_response(
        success=True,
        message="비디오 추론 API 준비 완료",
        data={"filename": file.filename}
    )
