"""FastAPI application factory."""

from fastapi import FastAPI

from app import __version__
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.lifespan import lifespan
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)
    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Research and screening-support API. Outputs are not diagnoses and "
            "require clinical review."
        ),
        lifespan=lifespan,
    )
    register_exception_handlers(application)
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_app()
