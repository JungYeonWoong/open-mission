from fastapi import APIRouter, UploadFile, File
from backend.utils.response import api_response
from backend.services.predict_service import PredictService

router = APIRouter()


@router.post("/image")
@router.post("/image/")
async def predict_image(file: UploadFile = File(...)):
    """
    이미지 업로드 → PredictService로 전체 추론 파이프라인 처리
    라우터는 URL 처리 및 응답 포맷만 담당
    """
    try:
        result = await PredictService.process_image(file)
        return api_response(
            success=True,
            message="이미지 추론 성공",
            data=result
        )
    except Exception as e:
        return api_response(
            success=False,
            message="이미지 추론 실패",
            error=str(e),
            data=None
        )


@router.post("/video")
async def predict_video(file: UploadFile = File(...)):
    """
    비디오 업로드 → PredictService로 전체 추론 파이프라인 처리
    """
    try:
        result = await PredictService.process_video(file)
        return api_response(
            success=True,
            message="비디오 추론 성공",
            data=result
        )
    except Exception as e:
        return api_response(
            success=False,
            message="비디오 추론 실패",
            error=str(e),
            data=None
        )
