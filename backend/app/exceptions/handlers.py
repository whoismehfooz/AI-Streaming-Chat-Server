"""
FastAPI exception handlers.

All expected application exceptions and framework-level HTTP errors are
translated into a predictable JSON response.

Backward-compatible response contract:
    {
        "detail": "A safe, human-readable message.",
        "error_code": "A_STABLE_MACHINE_READABLE_CODE"
    }

The ``detail`` field from the original handler is retained so existing
clients are not forced to change immediately. ``error_code`` is additive
and should be preferred by clients for programmatic error handling.

Request-validation responses additionally include a sanitized ``errors``
array. Raw request values, Pydantic input values, exception contexts,
tracebacks, and internal diagnostics are never returned to clients.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions.base import ApplicationException
from app.exceptions.error_codes import ErrorCode


logger = logging.getLogger(__name__)


def _error_code_value(error_code: ErrorCode) -> str:
    """Return the stable string value of an ErrorCode enum member."""
    return error_code.value


def _application_error_content(exc: ApplicationException) -> dict[str, Any]:
    """Build the backward-compatible public application-error body."""
    return {
        "detail": exc.message,
        "error_code": _error_code_value(exc.error_code),
    }


def _safe_validation_errors(
    validation_errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return validation locations and messages without echoing submitted data.

    Pydantic error dictionaries may contain ``input`` and ``ctx`` values.
    Those fields are deliberately omitted because they can contain
    passwords, tokens, personal data, or non-serializable objects.
    """
    sanitized: list[dict[str, Any]] = []

    for error in validation_errors:
        raw_location = error.get("loc", ())
        location: list[str | int] = []

        if isinstance(raw_location, (tuple, list)):
            for part in raw_location:
                if isinstance(part, bool):
                    location.append(str(part))
                elif isinstance(part, (str, int)):
                    location.append(part)
                else:
                    location.append(str(part)[:100])
        else:
            location.append(str(raw_location)[:100])

        raw_message = error.get("msg", "Invalid value.")
        raw_type = error.get("type", "value_error")

        message = (
            raw_message[:500]
            if isinstance(raw_message, str)
            else "Invalid value."
        )
        error_type = (
            raw_type[:150]
            if isinstance(raw_type, str)
            else "value_error"
        )

        sanitized.append(
            {
                "loc": location,
                "msg": message,
                "type": error_type,
            }
        )

    return sanitized


async def application_exception_handler(
    request: Request,
    exc: ApplicationException,
) -> JSONResponse:
    """
    Handle expected application exceptions.

    Client errors are not logged by default because they are expected
    control flow and can create noisy logs. Server-side application
    failures are logged with their traceback, while the response remains
    limited to the safe public message and stable error code.
    """
    if exc.status_code >= 500:
        logger.error(
            "Application exception during %s %s (error_code=%s)",
            request.method,
            request.url.path,
            _error_code_value(exc.error_code),
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=_application_error_content(exc),
        headers=exc.headers or None,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """
    Normalize FastAPI/Starlette HTTP exceptions.

    The HTTP exception's ``detail`` is an explicitly supplied public
    response value in Starlette's contract. String details are preserved.
    Non-string details are represented as a string to keep this response
    contract stable and JSON-safe. Framework-provided response headers,
    including WWW-Authenticate and Retry-After, are preserved.
    """
    # Preserve the original HTTPException detail shape (string, list,
    # mapping, etc.) while converting supported values to JSON-safe data.
    detail = jsonable_encoder(exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": detail,
            "error_code": ErrorCode.HTTP_ERROR.value,
        },
        headers=exc.headers,
    )


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle invalid request data without exposing submitted values.

    Only location, human-readable validation message, and error type are
    returned. The raw ``input`` and ``ctx`` entries from Pydantic are
    intentionally excluded.
    """
    validation_errors = _safe_validation_errors(exc.errors())

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed.",
            "error_code": ErrorCode.REQUEST_VALIDATION_ERROR.value,
            "errors": validation_errors,
        },
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Handle unexpected failures without leaking implementation details.

    The full exception is logged server-side. The client receives only a
    generic message and stable internal-server-error code.
    """
    logger.exception(
        "Unhandled application error during %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "error_code": ErrorCode.INTERNAL_SERVER_ERROR.value,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register the application's exception handlers.

    Keep this function as the single registration entry point used by
    main.py. Register the more specific exception types before Exception.
    """
    app.add_exception_handler(
        ApplicationException,
        application_exception_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        request_validation_exception_handler,
    )

    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )


__all__ = [
    "application_exception_handler",
    "http_exception_handler",
    "request_validation_exception_handler",
    "unhandled_exception_handler",
    "register_exception_handlers",
]
