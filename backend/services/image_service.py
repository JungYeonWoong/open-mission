import numpy as np
import cv2
from fastapi import UploadFile
from PIL import Image
from io import BytesIO
from pathlib import Path

class ImageService:
    """
    이미지 업로드 파일을 numpy 배열로 변환하는 기능을 담당하는 서비스 계층.
    """

    @staticmethod
    async def file_to_numpy(file: UploadFile) -> np.ndarray:
        """
        UploadFile -> numpy ndarray 로 변환하는 기능

        1. 바이트 데이터 읽기
        2. PIL.Image 로 로드
        3. numpy 배열로 변환
        4. BGR 형태로 변환 (YOLO/OpenCV에 맞춤)
        """
        # 1) 파일 바이트 읽기
        contents = await file.read()

        # 2) PIL 이미지 로드
        pil_img = Image.open(BytesIO(contents)).convert("RGB")

        # 3) numpy 배열로 변환
        img_np = np.array(pil_img)

        # 4) RGB → BGR (OpenCV, YOLO 추론 호환)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_np
    
    @staticmethod
    def save_result_image(img_bgr: np.ndarray, filename: str):
        """
        추론 결과 이미지를 backend/static/results/ 에 저장함.
        """
        save_dir = Path("backend/static/results")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"result_{filename}"
        cv2.imwrite(str(save_path), img_bgr)

        web_path = f"/static/results/result_{filename}"

        return web_path
