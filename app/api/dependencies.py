"""FastAPI dependency adapters for application services."""

from fastapi import Request

from app.core.errors import ModelNotReadyError
from app.services.la_screening import LAScreeningService
from app.services.screening import ScreeningService


def get_screening_service(request: Request) -> ScreeningService:
    """Return the lifespan-managed service or a stable readiness error."""
    service = getattr(request.app.state, "screening_service", None)
    if not isinstance(service, ScreeningService):
        raise ModelNotReadyError("The screening service has not started.")
    return service


def get_la_screening_service(request: Request) -> LAScreeningService:
    """Return the lifespan-managed LA service or a stable readiness error."""
    service = getattr(request.app.state, "la_screening_service", None)
    if not isinstance(service, LAScreeningService):
        raise ModelNotReadyError("The LA screening service has not started.")
    return service
