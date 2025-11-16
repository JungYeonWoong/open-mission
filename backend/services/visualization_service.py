import cv2
import numpy as np
from pathlib import Path

LABELS = {
    0: "Smoke_White",
    1: "Smoke_Grey",
    2: "Smoke_Black",
    3: "Fire",
}

COLORS = {
    0: (200, 200, 200),
    1: (128, 128, 128),
    2: (50, 50, 50),
    3: (0, 0, 255),
}


class VisualizationService:

    @staticmethod
    def draw_detections(img_bgr: np.ndarray, detections: np.ndarray):
        """YOLO 결과로 받은 박스들을 이미지에 그린다."""
        annotated = img_bgr.copy()

        for det in detections:
            VisualizationService._draw_single_detection(annotated, det)

        return annotated

    # ==========================
    # 🔹 Helper Functions
    # ==========================

    @staticmethod
    def _draw_single_detection(img: np.ndarray, det: np.ndarray):
        """단일 detection 데이터 처리."""
        x1, y1, x2, y2, conf, cls = VisualizationService._parse_det(det)
        color = COLORS.get(cls, (0, 0, 255))
        label = f"{LABELS.get(cls, cls)} {conf:.2f}"

        VisualizationService._draw_box(img, (x1, y1, x2, y2), color)
        VisualizationService._draw_label(img, (x1, y1), label, color)

    @staticmethod
    def _parse_det(det):
        """det array에서 값 파싱."""
        x1, y1, x2, y2, conf, cls = det
        return int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)

    @staticmethod
    def _draw_box(img, box, color):
        """바운딩 박스 그리기."""
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    @staticmethod
    def _draw_label(img, pos, text, color):
        """박스 위 텍스트 + 배경."""
        x1, y1 = pos
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
