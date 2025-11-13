# backend/tests/test_preprocess.py
"""
PreprocessService 전처리 기능 테스트
- letterbox padding
- RGB 변환
- normalize
- CHW 변환
"""

import sys
import numpy as np
from pathlib import Path

# backend 경로 추가
ROOT = Path(__file__).resolve().parents[2]  # open-mission/
sys.path.insert(0, str(ROOT))

from backend.services.preprocess_service import PreprocessService


def test_preprocess_image():
    print("\n==============================")
    print("🔥 PreprocessService 테스트 시작")
    print("==============================")

    # 1) 임의의 테스트 이미지 생성 (HWC)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 2) 전처리 실행
    processed = PreprocessService.preprocess_image(img, img_size=640)

    # 3) shape 확인
    print("📌 전처리 결과 shape:", processed.shape)
    assert processed.shape == (1, 3, 640, 640), "shape 불일치!"

    # 4) dtype 확인
    print("📌 dtype:", processed.dtype)
    assert processed.dtype == np.float32, "dtype이 float32가 아님!"

    # 5) 값 범위 확인
    print("📌 값 범위:", processed.min(), "~", processed.max())
    assert processed.min() >= 0.0 and processed.max() <= 1.0, "normalize 오류!"

    print("\n==============================")
    print("🎉 Preprocess 테스트 완료 — 정상 동작")
    print("==============================")


if __name__ == "__main__":
    test_preprocess_image()
