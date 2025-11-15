import cv2
import numpy as np
from pathlib import Path

class VisualizationService:

    @staticmethod
    def draw_detections(img_bgr: np.ndarray, detections: np.ndarray):
        """
        YOLO FireDetector.detect() 결과를 받아서 bounding boxes를 이미지에 그린다.
        detections: Nx6 (x1, y1, x2, y2, conf, cls)
        """
        annotated = img_bgr.copy()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.astype(int)

            # 박스
            cv2.rectangle(
                annotated, 
                (x1, y1), (x2, y2), 
                (0, 0, 255), 2
            )

            # 텍스트
            text = f"{int(cls)}:{conf:.2f}"
            cv2.putText(
                annotated, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2
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
import cv2
import numpy as np
from pathlib import Path

class VisualizationService:

    @staticmethod
    def draw_detections(img_bgr: np.ndarray, detections: np.ndarray):
        """
        YOLO FireDetector.detect() 결과를 받아서 bounding boxes를 이미지에 그린다.
        detections: Nx6 (x1, y1, x2, y2, conf, cls)
        """
        annotated = img_bgr.copy()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.astype(int)

            # 박스
            cv2.rectangle(
                annotated, 
                (x1, y1), (x2, y2), 
                (0, 0, 255), 2
            )

            # 텍스트
            text = f"{int(cls)}:{conf:.2f}"
            cv2.putText(
                annotated, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2
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
