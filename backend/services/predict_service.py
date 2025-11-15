import time
from fastapi import UploadFile
from pathlib import Path

from backend.services.image_service import ImageService
from backend.services.video_service import VideoService
from backend.services.preprocess_service import PreprocessService
from backend.services.inference_service import InferenceService
from backend.services.postprocess_service import PostprocessService
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

        # 3) 대표 프레임 추출
        frame = VideoService.extract_representative_frame(saved_path)
        
        # 4) 전처리
        processed = PreprocessService.preprocess_image(frame)

        # 추론
        start = time.time()
        raw_output = InferenceService.infer(processed)
        end = time.time()

        detections = PostprocessService.convert(raw_output)

        return {
            "filename": file.filename,
            "saved_path": saved_path,
            "frame_shape": frame.shape,
            "processed_size": processed.shape,
            "inference_time_ms": round((end - start) * 1000, 2),
            "detections": detections
        }
