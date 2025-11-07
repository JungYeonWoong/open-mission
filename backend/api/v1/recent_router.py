from fastapi import APIRouter
from backend.utils.response import api_response

router = APIRouter()

@router.get("/")
async def get_recent_list():
    # 나중에 실제 파일 목록 반환
    return api_response(
        success=True,
        message="최근 추론 목록 조회 성공",
        data=[]
    )

@router.get("/{item_id}")
async def get_recent_detail(item_id: str):
    return api_response(
        success=True,
        message=f"'{item_id}' 상세 조회 성공",
        data={"id": item_id}
    )
