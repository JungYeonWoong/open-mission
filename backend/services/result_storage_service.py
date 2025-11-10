import os
import json
import uuid
from datetime import datetime
from fastapi import UploadFile
from pathlib import Path
from PIL import Image
import numpy as np

RESULT_DIR = Path("backend/static/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


class ResultStorageService:
    """
    추론 결과(원본 이미지, 결과 이미지, 메타데이터)를 저장하는 서비스.
    """

    @staticmethod
    def _generate_id():
        """
        timestamp 기반 고유 ID 생성
        예: 20251122_153002_123456
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return now

    @staticmethod
    def _get_paths(result_id: str):
        """
        한 번의 추론 결과에 대한 모든 저장 경로 반환
        """
        return {
            "original": RESULT_DIR / f"{result_id}_original.png",
            "result": RESULT_DIR / f"{result_id}_result.png",
            "meta": RESULT_DIR / f"{result_id}_meta.json",
        }

    @staticmethod
    async def save_original_image(file: UploadFile, path: Path):
        """
        업로드한 원본 이미지를 PNG로 저장
        """
        data = await file.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        img.save(path, format="PNG")

    @staticmethod
    def save_result_image(img_np: np.ndarray, path: Path):
        """
        추론 결과 이미지(numpy)를 PNG로 저장
        (박스 렌더링된 이미지는 다음 단계에서 생성)
        """
        img = Image.fromarray(img_np.astype("uint8"))
        img.save(path, format="PNG")

    @staticmethod
    def save_meta(meta: dict, path: Path):
        """
        JSON 메타데이터 저장
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    async def save_all(file: UploadFile, original_np: np.ndarray, result_np: np.ndarray, meta: dict):
        """
        전체 저장 프로세스:
        원본 / 결과 이미지 / meta.json 모두 저장
        """
        result_id = ResultStorageService._generate_id()
        paths = ResultStorageService._get_paths(result_id)

        # 원본 이미지 저장
        ResultStorageService.save_result_image(original_np, paths["original"])

        # 결과 이미지 저장
        ResultStorageService.save_result_image(result_np, paths["result"])

        # 메타데이터 저장
        ResultStorageService.save_meta({
            "id": result_id,
            "filename": file.filename,
            **meta
        }, paths["meta"])

        return result_id
