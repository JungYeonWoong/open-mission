from fastapi import APIRouter, UploadFile, File
from backend.utils.response import api_response
from backend.services.image_service import ImageService
from backend.services.preprocess_service import PreprocessService

router = APIRouter()

@router.post("/image")
async def predict_image(file: UploadFile = File(...)):
    allowed_ext = ["jpg", "jpeg", "png"]
    ext = file.filename.split(".")[-1].lower()

    if ext not in allowed_ext:
        return api_response(
            success=False,
            message="지원하지 않는 파일 형식입니다.",
            error=f"Allowed: {allowed_ext}, Received: {ext}",
            data=None
        )

    try:
        # ndarray 변환
        img_np = await ImageService.file_to_numpy(file)

        # 전처리 (Letterbox + normalize + CHW)
        preprocessed = PreprocessService.preprocess_image(img_np)

    except Exception as e:
        return api_response(
            success=False,
            message="이미지 전처리 중 오류 발생",
            error=str(e),
            data=None
        )

    return api_response(
        success=True,
        message="이미지 변환 성공",
        data={
            "filename": file.filename,
            "shape": img_np.shape,  # 예: (720, 1280, 3)
            "processed_shape": preprocessed.shape
        }
    )

@router.post("/video")
async def predict_video(file: UploadFile = File(...)):
    return api_response(
        success=True,
        message="비디오 추론 API 준비 완료",
        data={"filename": file.filename}
    )
