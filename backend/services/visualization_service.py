import cv2
import numpy as np


class VisualizationService:
    """
    YOLO 결과(bbox, label, confidence)를 이미지 위에 그려주는 서비스.
    """

    @staticmethod
    def draw_boxes(image_np: np.ndarray, detections: list) -> np.ndarray:
        """
        원본 numpy 이미지 위에 YOLO 박스를 그린 후
        result.png로 저장할 수 있는 형태의 numpy 배열 반환.
        """
        img = image_np.copy()

        for det in detections:
            x1 = int(det["bbox"]["x1"])
            y1 = int(det["bbox"]["y1"])
            x2 = int(det["bbox"]["x2"])
            y2 = int(det["bbox"]["y2"])

            label = det["label"]
            conf = det["confidence"]

            # 색상 (클래스 ID 기반 색상)
            cls_id = det["class_id"]
            colors = [
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 165, 0),
                (255, 0, 255),
            ]
            color = colors[cls_id % len(colors)]

            # 1) 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 2) 라벨 텍스트
            text = f"{label} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 3) 텍스트 배경 박스
            cv2.rectangle(
                img,
                (x1, y1 - text_h - baseline),
                (x1 + text_w, y1),
                color,
                thickness=-1
            )

            # 4) 텍스트 그리기
            cv2.putText(
                img,
                text,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return img
