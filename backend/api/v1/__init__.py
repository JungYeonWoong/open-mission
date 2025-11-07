from fastapi import APIRouter

from .predict_router import router as predict_router
from .recent_router import router as recent_router

router = APIRouter()

router.include_router(predict_router, prefix="/predict", tags=["Predict"])
router.include_router(recent_router, prefix="/recent", tags=["Recent"])