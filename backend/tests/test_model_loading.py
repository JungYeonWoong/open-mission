"""
YOLO 모델 로딩 단독 테스트 스크립트
FastAPI 서버 없이 ModelLoader만 독립적으로 검증할 수 있다.
"""

import sys
from pathlib import Path

# backend 폴더를 Python 경로에 추가
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from backend.services.model_loader import ModelLoader


def test_model_loading():
    print("\n==============================")
    print("🔥 YOLO 모델 로딩 테스트 시작")
    print("==============================")

    model_path = ModelLoader._model_path
    print(f"📌 모델 경로 확인: {model_path}")

    if not model_path.exists():
        print("❌ 모델 파일이 존재하지 않습니다.")
        print("   → backend/models/ 폴더에 fire_detector.pt 파일이 있는지 확인하세요.")
        return

    # 모델 로딩
    model = ModelLoader.load_model()

    if model is None:
        print("❌ 모델 로딩 실패 (ModelLoader.load_model 반환값 = None)")
        return

    # 모델 summary 출력
    try:
        print("\n==============================")
        print("🧠 모델 Summary")
        print("==============================")
        print(model)
    except Exception as e:
        print(f"⚠️ Summary 출력 중 오류: {e}")

    print("\n==============================")
    print("✅ 테스트 완료 — 모델 로딩 정상 작동!")
    print("==============================")


if __name__ == "__main__":
    test_model_loading()
