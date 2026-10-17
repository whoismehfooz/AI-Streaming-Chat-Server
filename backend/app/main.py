from fastapi import FastAPI

from app.core.config import settings


from app.core.logging_config import configure_logging
from app.api import (
    chat_router,
    conversation_router,
    message_router
)
from app.exceptions.handlers import register_exception_handlers





app = FastAPI(
    title=settings.APP_NAME,
    description="Real time streaming AI chat server.",
    version="v1.0.0",
    contact={
        "name":"Mehfooz",
        "email":"mehfooz.sde@gmail.com"
    },
    license_info={
        "name":"MIT"
    },
    debug=settings.DEBUG
)



configure_logging()
app.include_router(chat_router),
app.include_router(conversation_router)
app.include_router(message_router)
register_exception_handlers(app)


@app.get("/health")
async def health_check():
    return {
        "status":"healthy",
        "service":settings.APP_NAME,
        "environment":settings.APP_ENV,
        "version":app.version
    }
