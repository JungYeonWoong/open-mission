import cv2
import numpy as np
from pathlib import Path

# YOLO Fire/Smoke 클래스 매핑
LABELS = {
    0: "Smoke_White",
    1: "Smoke_Grey",
    2: "Smoke_Black",
    3: "Fire"
}

# 클래스별 색상 (원하면 커스터마이징 가능)
COLORS = {
    0: (200, 200, 200),  # Smoke White - Light Gray
    1: (128, 128, 128),  # Smoke Grey - Gray
    2: (50, 50, 50),     # Smoke Black - Dark Gray
    3: (0, 0, 255)       # Fire - Red
}


class VisualizationService:

    @staticmethod
    def draw_detections(img_bgr: np.ndarray, detections: np.ndarray):
        """
        YOLO FireDetector.detect() 결과를 받아서 bounding boxes를 이미지에 그린다.
        detections: Nx6 (x1, y1, x2, y2, conf, cls)
        """
        annotated = img_bgr.copy()

        for det in detections:
            # det: [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2, conf, cls = det

            # 좌표는 정수 변환
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls = int(cls)
            conf = float(conf)

            label = LABELS.get(cls, str(cls))
            color = COLORS.get(cls, (0, 0, 255))  # default red

            # 박스 그리기
            cv2.rectangle(
                annotated,
                (x1, y1), (x2, y2),
                color,
                2
            )

            # 텍스트: "Fire 0.87" 형태
            text = f"{label} {conf:.2f}"

            # 텍스트 배경박스
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                color,
                -1
            )

            # 텍스트 그리기
            cv2.putText(
                annotated,
                text,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # white text
                2
            )

        return annotated


    @staticmethod
    def save_result_image(img_bgr: np.ndarray, filename: str):
        """
        추론 결과 이미지를 backend/static/results/ 에 저장함.
        """
        save_dir = Path("backend/static/results")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"result_{filename}"
        cv2.imwrite(str(save_path), img_bgr)

        return str(save_path)
