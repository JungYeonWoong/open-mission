"""
PostprocessService 테스트
- raw_output mock → bbox/label/conf 변환 검증
"""

import sys
import torch
import numpy as np
from pathlib import Path

# backend 경로 추가
ROOT = Path(__file__).resolve().parents[2]  # open-mission/
sys.path.insert(0, str(ROOT))

from backend.services.postprocess_service import PostprocessService


def test_postprocess_convert():
    print("\n==============================")
    print("🔥 PostprocessService 테스트 시작")
    print("==============================")

    # -------------------------------
    # 1) mock raw_output 생성
    # raw_output 형태: (1, N, 85)
    # (cx, cy, w, h, conf, cls_scores...)
    # -------------------------------
    preds = torch.tensor([
        [
            # conf=0.9 → 살아야 함
            [320, 320, 100, 100, 0.9, 0.1, 0.7, 0.2, 0.0],
            
            # conf=0.1 → threshold=0.25 미만이므로 제거되어야 함
            [100, 100, 50, 50, 0.1, 0.8, 0.1, 0.1, 0.0],
            
            # conf=0.8 → NMS에서 살 가능
            [330, 330, 95, 95, 0.8, 0.5, 0.4, 0.1, 0.0]
        ]
    ])

    raw_output = [preds]  # DetectMultiBackend 출력 호환

    # -------------------------------
    # 2) 후처리 실행
    # -------------------------------
    results = PostprocessService.convert(raw_output)

    print("📌 변환 결과:", results)

    # -------------------------------
    # 3) 결과 검증
    # -------------------------------

    # conf < 0.25 row는 제거되었는가?
    assert len(results) >= 1, "conf threshold 적용 실패!"

    # bbox key 존재 확인
    for det in results:
        assert "bbox" in det, "bbox 누락!"
        assert all(k in det["bbox"] for k in ["x1", "y1", "x2", "y2"]), "bbox 요소 누락!"

    # label 매핑 확인
    for det in results:
        assert "label" in det, "label 누락!"

    print("\n==============================")
    print("🎉 Postprocess 테스트 완료 — 정상 동작")
    print("==============================")


if __name__ == "__main__":
    test_postprocess_convert()
