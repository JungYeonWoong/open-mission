import time
import cv2
from fastapi import UploadFile
from pathlib import Path

from backend.services.image_service import ImageService
from backend.services.video_service import VideoService
from backend.services.preprocess_service import PreprocessService
# from backend.services.inference_service import InferenceService
# from backend.services.postprocess_service import PostprocessService
from backend.services.fire_detector import FireDetector
from backend.services.visualization_service import VisualizationService


class PredictService:
    """
    이미지/비디오 처리 전체 파이프라인을 담당하는 서비스 계층.
    라우터에서는 이 서비스만 호출하도록 구조를 단순화시킨다.
    """

    IMAGE_EXT = ["jpg", "jpeg", "png"]
    VIDEO_EXT = ["mp4", "mov", "avi"]

    @staticmethod
    def validate_extension(file: UploadFile, allowed_ext: list):
        ext = file.filename.split(".")[-1].lower()
        if ext not in allowed_ext:
            raise ValueError(f"지원하지 않는 파일 형식입니다. Allowed: {allowed_ext}, Received: '{ext}'")

    @staticmethod
    async def process_image(file: UploadFile):
        # 1) 확장자 검증
        PredictService.validate_extension(file, PredictService.IMAGE_EXT)

        # 2) ndarray 변환
        img_np = await ImageService.file_to_numpy(file)

        # 3) 전처리
        processed = PreprocessService.preprocess_image(img_np)

        # 4) 추론
        start = time.time()
        fire_detector = FireDetector("backend/models/fire.pt")
        detections = fire_detector.detect(img_np)
        end = time.time()

         # 5) 박스 그리기
        annotated = VisualizationService.draw_detections(img_np, detections)

        # 6) 이미지 저장
        saved_path = VisualizationService.save_result_image(annotated, file.filename)

        return {
            "filename": file.filename,
            "image_size": img_np.shape,
            "processed_size": processed.shape,
            "inference_time_ms": round((end - start) * 1000, 2),
            "detections": detections.tolist(),
            "saved_result_path": saved_path
        }

    @staticmethod
    async def process_video(file: UploadFile):
        # 1) 확장자 검증
        PredictService.validate_extension(file, PredictService.VIDEO_EXT)

        # 2) 비디오 저장
        saved_path = await VideoService.save_video(file)

        # 3) YOLO 로더
        fire_detector = FireDetector("backend/models/fire.pt")

        # 4) 비디오 읽기
        cap = cv2.VideoCapture(saved_path)
        if not cap.isOpened():
            raise RuntimeError("비디오를 열 수 없습니다.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FPS가 0인 경우 대비
        if not fps or fps < 1:
            fps = 25.0

        # 5) 출력 비디오 준비
        out_path = saved_path.replace(".mp4", "_result.mp4")

        # 🔥 먼저 H.264(avc1) 시도
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            print("⚠ avc1(H.264) 코덱 실패 → mp4v로 fallback")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # 그래도 안 열리면 오류
        if not writer.isOpened():
            raise RuntimeError("비디오 저장을 위한 VideoWriter를 열 수 없습니다.")

        # 6) 프레임 처리
        start = time.time()
        frame_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = fire_detector.detect(frame)
            annotated = VisualizationService.draw_detections(frame, detections)

            writer.write(annotated)
            frame_count += 1

        cap.release()
        writer.release()
        end = time.time()

        video_filename = Path(out_path).name
        web_video_path = f"/static/temp_videos/{video_filename}"

        return {
            "filename": file.filename,
            "saved_path": saved_path,
            "output_video": web_video_path,
            "total_frames": frame_count,
            "inference_time_ms": round((end - start) * 1000, 2)
        }
