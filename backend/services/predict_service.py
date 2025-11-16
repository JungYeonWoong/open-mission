import time
import cv2
from fastapi import UploadFile
from pathlib import Path

from backend.services.image_service import ImageService
from backend.services.video_service import VideoService
from backend.services.preprocess_service import PreprocessService
from backend.services.fire_detector import FireDetector
from backend.services.visualization_service import VisualizationService


class PredictService:
    """이미지/비디오 추론 파이프라인을 담당."""

    IMAGE_EXT = ["jpg", "jpeg", "png"]
    VIDEO_EXT = ["mp4", "mov", "avi"]
    fire_detector = FireDetector("backend/models/fire.pt")

    # ================================
    # 🔹 공용 유틸
    # ================================
    @staticmethod
    def validate_extension(file: UploadFile, allowed_ext: list):
        ext = file.filename.split(".")[-1].lower()
        if ext not in allowed_ext:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

    @staticmethod
    def _ensure_fps(fps):
        return fps if fps and fps > 1 else 25.0

    # ================================
    # 🔥 이미지 추론
    # ================================
    @staticmethod
    async def process_image(file: UploadFile):
        PredictService.validate_extension(file, PredictService.IMAGE_EXT)

        img_np = await ImageService.file_to_numpy(file)
        processed = PreprocessService.preprocess_image(img_np)

        start = time.time()
        detections = PredictService.fire_detector.detect(img_np)
        inference_ms = round((time.time() - start) * 1000, 2)

        annotated = VisualizationService.draw_detections(img_np, detections)
        saved_path = ImageService.save_result_image(annotated, file.filename)

        return {
            "filename": file.filename,
            "image_size": img_np.shape,
            "processed_size": processed.shape,
            "inference_time_ms": inference_ms,
            "detections": detections.tolist(),
            "saved_result_path": saved_path,
        }

    # ================================
    # 🔥 비디오 추론
    # ================================
    @staticmethod
    async def process_video(file: UploadFile):
        PredictService.validate_extension(file, PredictService.VIDEO_EXT)

        saved_path = await VideoService.save_video(file)
        cap = cv2.VideoCapture(saved_path)
        if not cap.isOpened():
            raise RuntimeError("비디오를 열 수 없습니다.")

        fps = PredictService._ensure_fps(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = saved_path.replace(".mp4", "_result.mp4")
        writer = PredictService._init_writer(out_path, fps, w, h)

        start = time.time()
        frame_count = PredictService._process_frames(cap, writer)
        inference_ms = round((time.time() - start) * 1000, 2)

        cap.release()
        writer.release()
        video_filename = Path(out_path).name
        web_video_path = f"/static/temp_videos/{video_filename}"

        return {
            "filename": file.filename,
            "saved_path": saved_path,
            "output_video": web_video_path,
            "total_frames": frame_count,
            "inference_time_ms": inference_ms,
        }

    # ================================
    # 🔹 비디오 보조 함수들 (적당히 분리)
    # ================================
    @staticmethod
    def _init_writer(out_path, fps, w, h):
        """브라우저 호환성 높은 H.264 우선 생성"""
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            print("⚠ avc1 실패 → mp4v로 fallback")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError("VideoWriter 생성 실패")

        return writer

    @staticmethod
    def _process_frames(cap, writer):
        """프레임 반복 처리 및 박스 시각화"""
        frame_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = PredictService.fire_detector.detect(frame)
            annotated = VisualizationService.draw_detections(frame, detections)
            writer.write(annotated)
            frame_count += 1

        return frame_count
