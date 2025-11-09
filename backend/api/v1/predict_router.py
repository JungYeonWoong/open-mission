from fastapi import APIRouter, UploadFile, File
from backend.utils.response import api_response
from backend.services.predict_service import PredictService

router = APIRouter()


@router.post("/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        result = await PredictService.process_image(file)

        return api_response(
            success=True,
            message="이미지 처리 성공",
            data=result
        )

    except Exception as e:
        return api_response(
            success=False,
            message="이미지 처리 실패",
            error=str(e),
            data=None
        )


@router.post("/video")
async def predict_video(file: UploadFile = File(...)):
    try:
        result = await PredictService.process_video(file)

        return api_response(
            success=True,
            message="비디오 처리 성공",
            data=result
        )

    except Exception as e:
        return api_response(
            success=False,
            message="비디오 처리 실패",
            error=str(e),
            data=None
        )
