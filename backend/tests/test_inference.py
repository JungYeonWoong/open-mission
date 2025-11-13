# backend/tests/test_inference.py
"""
YOLOv5 모델 Inference 단독 테스트
- 전처리/후처리 없이 모델 forward()만 검증
"""

import sys
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # open-mission/
sys.path.insert(0, str(ROOT))

# backend 경로 등록
#ROOT = Path(__file__).resolve().parents[1]
#sys.path.insert(0, str(ROOT))

from backend.services.model_loader import ModelLoader


def test_inference():
    print("\n==============================")
    print("🔥 YOLO 모델 Inference 단독 테스트 시작")
    print("==============================")

    # 1) 모델 로딩
    model = ModelLoader.load_model()
    if model is None:
        print("❌ 모델 로딩 실패 (ModelLoader.load_model 반환값이 None)")
        return

    print("✅ 모델 로딩 완료")

    # 2) dummy 입력 생성 (YOLOv5의 기본 입력 크기)
    dummy_np = np.zeros((1, 3, 640, 640), dtype=np.float32)
    dummy_tensor = torch.from_numpy(dummy_np)

    # 3) device 일치시키기
    device = next(model.parameters()).device
    dummy_tensor = dummy_tensor.to(device)

    # 4) forward 수행
    try:
        with torch.no_grad():
            raw_output = model(dummy_tensor)

        print("✅ forward() 실행 성공")

    except Exception as e:
        print(f"❌ forward 중 오류 발생: {e}")
        return

    # 5) 출력 구조 검증
    print("\n==============================")
    print("🧪 출력 구조 검증")
    print("==============================")

    if isinstance(raw_output, (list, tuple)):
        print("📌 raw_output 타입:", type(raw_output))
        tensor = raw_output[0]
    else:
        tensor = raw_output

    print("📌 출력 텐서 shape:", tensor.shape)

    # 보통 YOLOv5: (1, N, 85)
    if tensor.dim() == 3 and tensor.shape[-1] >= 6:
        print("✅ 출력 텐서 형태 정상")
    else:
        print("⚠️ 출력 형태가 예상과 다릅니다. 후처리 단계에서 오류가 날 수 있습니다.")

    print("\n==============================")
    print(" Inference 단독 테스트 완료")
    print("==============================")


if __name__ == "__main__":
    test_inference()
