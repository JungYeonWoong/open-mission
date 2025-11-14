"""
실제 이미지 → 실제 전처리 → 실제 모델 → 실제 후처리
REAL PredictService end-to-end 테스트
"""

import sys
import numpy as np
import torch
from pathlib import Path
from starlette.datastructures import UploadFile
from io import BytesIO
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.services.predict_service import PredictService
from backend.services.model_loader import ModelLoader


def test_predict_real_image():
    print("\n==============================")
    print("🔥 REAL PredictService 이미지 테스트 시작")
    print("==============================")

    # --------------------------------------
    # 1) 실제 이미지 파일 로드
    # --------------------------------------
    image_path = ROOT / "backend" / "tests" / "sample" / "test_img.jpg"
    assert image_path.exists(), f"테스트 이미지가 존재하지 않음: {image_path}"

    # UploadFile mock 생성 (실제 이미지 파일)
    with open(image_path, "rb") as f:
        data = f.read()
    upload = UploadFile(filename="test_image.jpg", file=BytesIO(data))

    # --------------------------------------
    # 2) 실제 모델 로딩
    # --------------------------------------
    model = ModelLoader.load_model()
    assert model is not None, "❌ 모델 로딩 실패 — fire_detector.pt 확인 필요!"
    print("✅ 실제 YOLO 모델 로딩 성공")

    # --------------------------------------
    # 3) PredictService 호출
    # --------------------------------------
    import asyncio
    result = asyncio.run(PredictService.process_image(upload))

    print("📌 Predict 결과:", result)

    # --------------------------------------
    # 4) 결과 검증
    # --------------------------------------
    assert "filename" in result
    assert "image_size" in result
    assert "processed_size" in result
    assert "detections" in result

    assert isinstance(result["detections"], list), "detections는 list여야 함!"
    print(f"🔍 감지 결과 {len(result['detections'])}개")

    print("\n==============================")
    print("🎉 REAL PredictService 테스트 완료 — 정상 동작")
    print("==============================")


if __name__ == "__main__":
    test_predict_real_image()
