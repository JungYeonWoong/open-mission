import torch
from pathlib import Path
import sys

# ======================================
# YOLOv5 경로 설정 (전역에서 한 번만 설정)
# ======================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # open-mission/
YOLO_ROOT = ROOT / "yolov5"

# sys.path 최우선 등록
sys.path.insert(0, str(YOLO_ROOT))
sys.path.insert(0, str(YOLO_ROOT / "utils"))

from yolov5.models.common import DetectMultiBackend


class ModelLoader:
    """
    YOLO 모델 로딩 + warm-up + 예외 처리 강화 버전.
    예외 발생 시 모델은 None 상태로 유지하며,
    predict 단계에서 모델 존재 여부를 확인하여 적절한 응답을 반환하게 된다.
    """

    _model = None
    _model_path = (ROOT / "backend" / "models" / "fire_detector.pt").resolve()

    @classmethod
    def load_model(cls):
        """안전한 예외 처리를 포함하는 모델 로딩 메서드"""
        if cls._model is not None:
            return cls._model

        # YOLOv5 폴더 등록

        # 모델 파일 존재 여부 확인
        if not cls._model_path.exists():
            print(f"[ModelLoader] 모델 파일이 존재하지 않습니다 → {cls._model_path}")
            cls._model = None
            return None

        try:
            # YOLOv5 모델 로딩
            cls._model = DetectMultiBackend(
                weights=str(cls._model_path),
                device="cpu",    # GPU 사용 시 'cuda:0'
                dnn=False
            )
            print("[ModelLoader] 모델 로딩 성공!")

        except Exception as e:
            print(f"[ModelLoader] 모델 로딩 실패: {e}")
            cls._model = None
            return None

        # warm-up 실행 (실패해도 서버는 계속 실행되도록 soft fail)
        try:
            cls.warm_up()
        except Exception as e:
            print(f"[ModelLoader] warm-up 실패: {e}")


        return cls._model

    @classmethod
    def warm_up(cls):
        """dummy 입력 기반 warm-up"""
        if cls._model is None:
            return

        dummy_input = torch.zeros((1, 3, 640, 640))
        cls._model(dummy_input)  # warm-up inference

    @classmethod
    def get_model(cls):
        """로드된 모델 반환"""
        return cls._model
