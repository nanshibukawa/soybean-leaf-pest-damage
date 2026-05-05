from fastapi import APIRouter
from app import endpoint

router = APIRouter()
router.include_router(endpoint.router, prefix="/events", tags=["events"])
