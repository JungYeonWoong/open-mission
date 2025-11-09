import cv2
import numpy as np
from typing import Tuple


class PreprocessService:
    """
    YOLO 모델 입력을 위한 전처리 기능을 담당하는 서비스 계층.
    - Resize + Letterbox padding
    - BGR → RGB
    - Normalize
    - CHW 변환
    """

    @staticmethod
    def letterbox(
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
    ):
        """
        YOLOv5 스타일의 letterbox padding 적용
        """
        shape = img.shape[:2]  # (h, w)

        # scale 계산
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        # Resize
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Padding 계산
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # Padding 적용
        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=color
        )

        return padded

    @staticmethod
    def preprocess_image(img: np.ndarray, img_size: int = 640) -> np.ndarray:
        """
        YOLO 모델 입력에 맞게 이미지 전처리 수행.
        """
        # Letterbox padding 적용
        img = PreprocessService.letterbox(img, (img_size, img_size))

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # float32 & normalize
        img = img.astype(np.float32) / 255.0

        # CHW 변환 (HWC → CHW)
        img = np.transpose(img, (2, 0, 1))

        # Batch dimension 추가 → (1, 3, 640, 640)
        img = np.expand_dims(img, axis=0)

        return img
