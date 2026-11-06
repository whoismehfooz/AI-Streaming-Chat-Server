import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions.custom_exceptions import ApplicationException


logger = logging.getLogger(__name__)


async def application_exception_handler(
    request: Request,
    exc: ApplicationException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": str(exc),
        },
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled application error during %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        ApplicationException,
        application_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )
