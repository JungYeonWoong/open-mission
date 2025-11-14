from ultralytics import YOLO
from rbln_sdk import Compiler

# 1️⃣ PyTorch YOLOv5 모델 불러오기
model = YOLO("daon_fire4.pt")

# 2️⃣ (선택) ONNX로 변환 - 디버깅 및 확인용
onnx_path = model.export(format="onnx", imgsz=640)

# 3️⃣ RBLN SDK 컴파일러로 변환
compiler = Compiler()

# PyTorch 혹은 ONNX 중 하나 선택
compiler.load_model(onnx_path, framework="onnx", input_shape=(1, 3, 640, 640))
# compiler.load_model("yolov5s.pt", framework="pytorch", input_shape=(1, 3, 640, 640))

# 4️⃣ NPU 최적화 및 양자화 옵션 설정
compiler.compile(
    target="rbln_npu",
        quantize="int8",        # 정수 양자화 (성능 향상)
	    optimize=True,          # 최적화 적용
	        batch_size=1
		)

# 5️⃣ 변환된 모델 저장
compiler.save("daon_fire4_rbln.bin")

print("✅ RBLN 변환 완료: yolov5s_rbln.bin 생성됨")

