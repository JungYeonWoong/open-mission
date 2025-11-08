import torch
from pathlib import Path


class ModelLoader:
    """
    YOLO 모델을 메모리에 로딩하고 warm-up을 수행하는 모듈.
    예외 처리 강화는 다음 커밋(refactor)에서 진행한다.
    """

    _model = None
    _model_path = Path("backend/models/fire_detector.pt")

    @classmethod
    def load_model(cls):
        """모델을 메모리에 로드"""
        if cls._model is not None:
            return cls._model

        cls._model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(cls._model_path),
            source="github"
        )

        # 모델 로딩 후 warm-up 실행
        cls.warm_up()

        return cls._model

    @classmethod
    def warm_up(cls):
        """
        초기 추론 속도를 빠르게 하기 위해 dummy input으로 warm-up 수행.
        """
        if cls._model is None:
            return  # 모델이 아직 없는 경우 skip

        # YOLOv5의 기본 입력 크기 기준 dummy tensor 생성
        dummy_input = torch.zeros((1, 3, 640, 640))

        # Warm-up inference
        cls._model(dummy_input)

    @classmethod
    def get_model(cls):
        """로드된 모델 반환"""
        return cls._model
