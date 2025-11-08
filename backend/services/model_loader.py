import torch
from pathlib import Path


class ModelLoader:
    """
    YOLO 모델을 메모리에 로딩하는 기본 모듈.
    warm-up 기능 및 예외 처리 강화는 이후 커밋에서 별도로 추가한다.
    """

    _model = None
    _model_path = Path("backend/models/fire_detector.pt")  

    @classmethod
    def load_model(cls):
        """모델을 메모리에 로드하는 가장 기본적인 기능"""
        # 이미 로드된 경우 반환
        if cls._model is not None:
            return cls._model

        # 모델 파일 경로를 그대로 사용 
        cls._model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(cls._model_path),
            source="github",
        )

        return cls._model

    @classmethod
    def get_model(cls):
        """로드된 모델 인스턴스 반환"""
        return cls._model
