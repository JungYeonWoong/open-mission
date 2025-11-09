import shutil
import cv2
import numpy as np
from fastapi import UploadFile
from pathlib import Path


class VideoService:
    """
    비디오 업로드 및 대표 프레임 추출 기능을 담당하는 서비스 계층.
    """

    TEMP_DIR = Path("backend/static/temp_videos")

    @staticmethod
    async def save_video(file: UploadFile) -> str:
        """업로드된 비디오 파일을 임시 디렉토리에 저장"""
        VideoService.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        save_path = VideoService.TEMP_DIR / file.filename

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return str(save_path)

    @staticmethod
    def extract_representative_frame(video_path: str) -> np.ndarray:
        """
        비디오 파일에서 대표 프레임을 추출하여 numpy 배열 형태로 반환한다.
        기본적으로 첫 번째 프레임을 추출한다.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            raise RuntimeError("비디오 프레임을 읽을 수 없습니다.")

        return frame  # BGR 형태 (전처리는 다음 단계에서 적용)
