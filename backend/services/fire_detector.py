import cv2
import torch
import numpy as np
from pathlib import Path


class FireDetector:
    """
    Slim FireDetector
    - torch.hub.load 로 YOLOv5 모델 정상 로딩
    - 이미지 → RGB → 모델 호출 → xyxy 결과 반환
    - 단일 모델만 사용
    """

    def __init__(self, model_path: str, device: str = "cpu", conf=0.25, iou=0.45):
        self.model_path = str(Path(model_path).resolve())
        self.device = torch.device(device)

        # YOLOv5 모델 로드
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=self.model_path,
            source="github",   # ← 이걸로 강제
            force_reload=False,
            verbose=False
        )
        # 설정
        self.model.to(self.device)
        self.model.conf = conf
        self.model.iou = iou
        self.model.eval()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """YOLOv5는 BGR→RGB만 해주면 이미지 바로 넣어도 됨."""
        rgb = frame[:, :, ::-1]  # BGR → RGB
        return rgb

    def detect(self, frame: np.ndarray):
        """
        frame (BGR) → YOLOv5 inference → xyxy 결과 반환
        return: numpy array [N, 6] (x1,y1,x2,y2,conf,cls)
        """
        rgb = self._preprocess(frame)

        with torch.no_grad():
            results = self.model(rgb)
            detections = results.xyxy[0].cpu().numpy()

        return detections
