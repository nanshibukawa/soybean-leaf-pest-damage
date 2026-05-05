import json
from http import HTTPStatus
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import Response

router = APIRouter()


class EventSchema(BaseModel):
    event_id: str
    event_type: str
    event_data: dict


@router.post("/", dependencies=[])
def handle_event(data: EventSchema) -> Response:
    print(data)
    return Response(
        content=json.dumps({"message": "Data received!"}),
        status_code=HTTPStatus.ACCEPTED,
    )
