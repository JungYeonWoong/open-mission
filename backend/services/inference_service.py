import torch
import numpy as np

from backend.services.model_loader import ModelLoader


class InferenceService:
    """
    YOLO 모델에 대한 순수 추론(Inference)만 담당하는 서비스.
    """

    @staticmethod
    def _numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
        """
        전처리된 numpy(CHW, float32, 0~1) → torch tensor(batch 포함)
        """
        # numpy (1,3,640,640) 또는 (3,640,640)
        if img.ndim == 3:
            img = np.expand_dims(img, 0)  # (1,3,H,W)
        return torch.from_numpy(img).float()

    @staticmethod
    def infer(img):
        """
        YOLO 모델 추론
        numpy 또는 torch.Tensor 입력 모두 지원
        """
        model = ModelLoader.get_model()
        if model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. (ModelLoader.load_model 필요)")
        
        model = model.float() 

        # numpy 입력이면 자동 변환
        if isinstance(img, np.ndarray):
            img = InferenceService._numpy_to_tensor(img)

        # torch.Tensor 보장됨
        device = next(model.parameters()).device
        img = img.to(device)

        # forward
        with torch.no_grad():
            raw_outputs = model(img)

        return raw_outputs
