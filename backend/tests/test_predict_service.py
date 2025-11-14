import sys
import numpy as np
import torch
from pathlib import Path
from io import BytesIO
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from backend.services.predict_service import PredictService
from backend.services.inference_service import InferenceService


# -----------------------------
# 🔥 torch 기반 Mock YOLO raw_output 생성
# -----------------------------
mock_raw_output = [
    torch.tensor([  # batch dimension
        [   # these are detections (N rows)
            [320, 320, 100, 100, 0.9, 0.1, 0.8, 0.1],
            [120, 120,  50,  50, 0.3, 0.6, 0.2, 0.2],
        ]
    ])
]


def mock_infer(_):
    return mock_raw_output


def test_predict_service_image():
    print("\n==============================")
    print("🖼 PredictService 이미지 테스트 시작")
    print("==============================")

    # 1) UploadFile mock 이미지 생성
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    from PIL import Image
    img = Image.fromarray(rgb, "RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    file = UploadFile(filename="test.png", file=buffer)

    # 2) InferenceService.infer mock 적용
    InferenceService.infer = mock_infer   # torch Tensor 반환

    # 3) PredictService 실행
    import asyncio
    result = asyncio.run(PredictService.process_image(file))

    print("📌 Predict 결과:", result)

    # 4) 구조 검증
    assert "filename" in result
    assert "image_size" in result
    assert "processed_size" in result
    assert "detections" in result
    assert isinstance(result["detections"], list)

    print("\n==============================")
    print("🎉 PredictService 테스트 완료 — 정상 동작")
    print("==============================")


if __name__ == "__main__":
    test_predict_service_image()
