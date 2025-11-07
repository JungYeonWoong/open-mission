from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/image")
async def predict_image(file: UploadFile = File(...)):
    # 추론 기능 구현 전, 기본 형태만 작성
    return {"message": "predict image endpoint placeholder"}

@router.post("/video")
async def predict_video(file: UploadFile = File(...)):
    return {"message": "predict video endpoint placeholder"}
