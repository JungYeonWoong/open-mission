import cv2
import numpy as np
from rbln_sdk import Runtime

# 1️⃣ 런타임 초기화
runtime = Runtime(device="rbln_npu0")

# 2️⃣ 모델 로드
runtime.load_model("yolov5s_rbln.bin")

# 3️⃣ 입력 이미지 전처리
img = cv2.imread("data/images/bus.jpg")  # 테스트용 이미지
img_resized = cv2.resize(img, (640, 640))
img_norm = img_resized.astype(np.float32) / 255.0
img_input = np.transpose(img_norm, (2, 0, 1))  # HWC → CHW
img_input = np.expand_dims(img_input, 0)  # 배치 차원 추가

# 4️⃣ NPU 추론 실행
outputs = runtime.infer(inputs=[img_input])
print("✅ 추론 완료")

# 5️⃣ 후처리 (YOLO의 NMS 예시)
def nms(boxes, scores, iou_thresh=0.5):
    # 단순 NMS 구현 (예시)
        keep = []
	    idxs = np.argsort(scores)[::-1]
	        while len(idxs) > 0:
			        i = idxs[0]
				        keep.append(i)
					        iou = np.sum(np.minimum(boxes[i], boxes[idxs[1:]])) / np.sum(np.maximum(boxes[i], boxes[idxs[1:]]))
						        idxs = idxs[np.where(iou < iou_thresh)[0] + 1]
							    return keep

# 결과 확인
for det in outputs:
	    print(det)
	    x
