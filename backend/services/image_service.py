from fastapi import UploadFile
from typing import Optional


class ImageService:
    """
    이미지 처리에 대한 비즈니스 로직을 담당하는 서비스 계층.
    아직 numpy 변환이나 전처리는 다음 커밋에서 추가한다.
    """

    @staticmethod
    async def handle_image_upload(file: UploadFile) -> Optional[str]:
        """
        이미지 UploadFile을 처리하기 위한 기본 함수.
        현재는 파일 이름만 반환하는 placeholder 역할만 수행한다.
        """
        return file.filename
