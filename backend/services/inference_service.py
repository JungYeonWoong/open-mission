import torch
import numpy as np

from backend.services.model_loader import ModelLoader


class InferenceService:
    """
    YOLO 모델에 대한 순수 추론(Inference)만 담당하는 서비스.
    - 후처리(bbox 계산, NMS)는 다음 단계에서 분리하여 구현한다.
    """

    @staticmethod
    def _numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
        """
        전처리된 numpy(CHW, float32, 0~1) → torch tensor(batch 포함)
        """
        if img.ndim == 3:
            img = np.expand_dims(img, 0)   # (1,3,640,640)
        return torch.from_numpy(img).float()

    @staticmethod
    def infer(img_tensor: torch.Tensor):
        """
        YOLO 모델 추론 (raw output만 반환)
        """
        model = ModelLoader.get_model()
        if model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. (ModelLoader.load_model 필요)")

        # device 자동 선택
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # 모델 forward
        with torch.no_grad():
            raw_outputs = model(img_tensor)

        return raw_outputs
