from fastapi import APIRouter

from ..schemas import HealthResponse


class HealthRoutes:
    def __init__(self) -> None:
        self.router = APIRouter(prefix="/api/v1", tags=["health"])
        self.router.add_api_route("/health", self.health, methods=["GET"], response_model=HealthResponse)

    @staticmethod
    def health() -> HealthResponse:
        return HealthResponse(status="ok")