from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions.custom_exceptions import (
    AIProviderException,
    ConversationNotFoundException,
    ContextBudgetExceededException
)


def register_exception_handlers(app:FastAPI):
    @app.exception_handler(AIProviderException)
    async def ai_provider_exception_handler(
        request: Request,
        exc: AIProviderException
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail":str(exc)
            }
        )

    @app.exception_handler(ConversationNotFoundException)
    async def conversation_not_found_exc_handler(
        request: Request,
        exc: ConversationNotFoundException
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail":str(exc)
            }
        )

    @app.exception_handler(ContextBudgetExceededException)
    async def context_budget_exc_hanlder(
        request: Request,
        exc: ContextBudgetExceededException
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail":str(exc)
            }
        )
