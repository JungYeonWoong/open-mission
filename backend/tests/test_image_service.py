# backend/tests/test_image_service.py
"""
ImageService 테스트
- UploadFile → numpy BGR 이미지 변환 검증
"""

import sys
import numpy as np
from pathlib import Path
from io import BytesIO
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.services.image_service import ImageService


def test_image_service_file_to_numpy():
    print("\n==============================")
    print("🔥 ImageService 테스트 시작")
    print("==============================")

    # -----------------------------------------
    # 1) 임의의 RGB 테스트 이미지 생성 (PIL이 읽을 수 있도록)
    # -----------------------------------------
    # (H, W, 3)
    rgb_array = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb_array[:, :, 0] = 255  # red 채널

    # numpy → PNG 파일 형태로 BytesIO에 저장
    from PIL import Image
    img = Image.fromarray(rgb_array, "RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # UploadFile mock 생성
    upload = UploadFile(filename="test.png", file=buffer)

    # -----------------------------------------
    # 2) ImageService 실행
    # -----------------------------------------
    img_np = None
    import asyncio
    img_np = asyncio.run(ImageService.file_to_numpy(upload))

    print("📌 변환된 numpy shape:", img_np.shape)
    print("📌 dtype:", img_np.dtype)

    # -----------------------------------------
    # 3) 검증
    # -----------------------------------------
    # shape 확인
    assert img_np.shape == (100, 200, 3), "shape이 원본과 다름!"

    # dtype 확인
    assert img_np.dtype == np.uint8, "dtype이 uint8이 아님!"

    # RGB → BGR 변환 확인
    # 원래 red(255,0,0)이었던 픽셀 → BGR에서는 (0,0,255)이어야 함
    assert img_np[0, 0, 2] == 255, "RGB→BGR 변환 실패!"
    assert img_np[0, 0, 0] == 0, "BGR 변환 값 오류!"

    print("\n==============================")
    print("🎉 ImageService 테스트 완료 — 정상 동작")
    print("==============================")


if __name__ == "__main__":
    test_image_service_file_to_numpy()
