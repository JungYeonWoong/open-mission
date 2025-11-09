# backend/services/postprocess_service.py

import torch
import numpy as np


class PostprocessService:
    """
    YOLO 모델의 raw output을 바운딩박스/라벨/신뢰도로 변환하는 후처리 단계.
    - NMS 수행
    - 박스 좌표 디코딩
    - confidence threshold 적용
    - label 매핑
    """

    CONF_THRES = 0.25
    IOU_THRES = 0.45

    # YOLOv5 클래스 라벨
    LABELS = [
        "smoke1", "smoke2", "smoke3", "Flame",
        # 필요한 label 넣으면 됨
    ]

    @staticmethod
    def _xywh_to_xyxy(box):
        """
        YOLO 형식의 (cx,cy,w,h) → (x1,y1,x2,y2)
        """
        x_c, y_c, w, h = box
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return [x1, y1, x2, y2]

    @staticmethod
    def apply_nms(predictions):
        """
        predictions: YOLO raw output (tensor)
        returns: NMS 적용된 tensor
        """
        return torch.nn.functional.nms(
            predictions[:, :4],
            predictions[:, 4] * predictions[:, 5:].max(1)[0],
            PostprocessService.IOU_THRES
        )

    @staticmethod
    def convert(raw_output):
        """
        YOLO raw_output → JSON friendly list of detections
        
        raw_output shape: (batch, N, 85)
        각 row = [cx, cy, w, h, conf, class_scores ...]
        """

        if isinstance(raw_output, list):  # DetectMultiBackend 호환
            raw_output = raw_output[0]

        preds = raw_output[0]  # batch=1 가정
        preds = preds.cpu()

        results = []

        # confidence threshold
        mask = preds[:, 4] > PostprocessService.CONF_THRES
        preds = preds[mask]

        if preds.shape[0] == 0:
            return []

        # class confidence + id
        class_conf, class_ids = preds[:, 5:].max(1)

        # final confidence = obj_conf * class_conf
        conf = preds[:, 4] * class_conf

        # NMS 수행
        boxes = preds[:, :4]
        nms_idx = torch.nn.functional.nms(boxes, conf, PostprocessService.IOU_THRES)

        preds = preds[nms_idx]
        class_ids = class_ids[nms_idx]
        conf = conf[nms_idx]

        for i in range(len(preds)):
            xyxy = PostprocessService._xywh_to_xyxy(preds[i][:4].numpy())
            cls_id = int(class_ids[i])
            results.append({
                "class_id": cls_id,
                "label": PostprocessService.LABELS[cls_id] if cls_id < len(PostprocessService.LABELS) else f"class_{cls_id}",
                "confidence": float(conf[i]),
                "bbox": {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3])
                }
            })

        return results
