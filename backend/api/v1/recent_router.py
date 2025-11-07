from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_recent_list():
    return {"message": "recent history list placeholder"}

@router.get("/{item_id}")
async def get_recent_detail(item_id: str):
    return {"message": f"recent detail placeholder for {item_id}"}
