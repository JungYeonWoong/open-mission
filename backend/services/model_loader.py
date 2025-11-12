import torch
from pathlib import Path
from models.common import DetectMultiBackend

class ModelLoader:
    """
    YOLO 모델 로딩 + warm-up + 예외 처리 강화 버전.
    예외 발생 시 모델은 None 상태로 유지하며,
    predict 단계에서 모델 존재 여부를 확인하여 적절한 응답을 반환하게 된다.
    """

    _model = None
    _model_path = Path("backend/models/fire_detector.pt")

    @classmethod
    def load_model(cls):
        """안전한 예외 처리를 포함하는 모델 로딩 메서드"""
        if cls._model is not None:
            return cls._model

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
