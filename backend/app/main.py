import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.firebase import initialize_firebase
from app.exception_handlers import register_exception_handlers
from app.routers import health_router, auth_router, document_router, chat_router, conversation_router

# Initialize logging system
setup_logging("DEBUG" if settings.ENVIRONMENT == "development" else "INFO")
logger = logging.getLogger("app.main")

# Instantiate FastAPI Application
app = FastAPI(
    title="PDF Chatbot API",
    description="FastAPI Backend for PDF RAG Chatbot",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Request body size limiting at ASGI/server level (15MB max)
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi.responses import JSONResponse

class LimitRequestSizeMiddleware:
    def __init__(self, app: ASGIApp, max_content_size: int = 15 * 1024 * 1024):
        self.app = app
        self.max_content_size = max_content_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            content_length = 0
            for header_name, header_value in scope.get("headers", []):
                if header_name == b"content-length":
                    try:
                        content_length = int(header_value)
                    except ValueError:
                        pass
                    break
            
            if content_length > self.max_content_size:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "success": False,
                        "message": "Request entity too large",
                        "errors": [{"detail": f"Request size exceeds limit of {self.max_content_size} bytes."}]
                    }
                )
                await response(scope, receive, send)
                return

            bytes_received = 0
            async def wrapped_receive() -> dict:
                nonlocal bytes_received
                message = await receive()
                if message["type"] == "http.request":
                    body_len = len(message.get("body", b""))
                    bytes_received += body_len
                    if bytes_received > self.max_content_size:
                        raise ValueError("Request body size limit exceeded")
                return message

            try:
                await self.app(scope, wrapped_receive, send)
                return
            except ValueError as e:
                if str(e) == "Request body size limit exceeded":
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "success": False,
                            "message": "Request entity too large",
                            "errors": [{"detail": "Streamed request body exceeded limit."}]
                        }
                    )
                    await response(scope, receive, send)
                    return
                raise e
        
        await self.app(scope, receive, send)

# Register request size limiter at server level
app.add_middleware(LimitRequestSizeMiddleware, max_content_size=15 * 1024 * 1024)

# CORS configuration (strictly explicit, no wildcard allowed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exceptions handlers mapping to envelope format
register_exception_handlers(app)

# Include routers (API Version prefix applied globally)
app.include_router(health_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(document_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(conversation_router, prefix="/api/v1")


# OLD: startup event initializing only Firebase SDK — replaced below to add local Ollama server connection check during startup
# @app.on_event("startup")
# async def startup_event() -> None:
#     # Initialize Firebase Admin SDK
#     initialize_firebase()
#     logger.info(f"PDF Chatbot backend started in environment: '{settings.ENVIRONMENT}'")
#     logger.info(f"CORS origins configured: {settings.cors_origins_list}")

@app.on_event("startup")
async def startup_event() -> None:
    # Initialize Firebase Admin SDK
    initialize_firebase()
    
    # Pre-warm local embedding model in the background so it doesn't block startup or health check
    if settings.EMBEDDING_PROVIDER == "local":
        import asyncio
        from app.services.embedding_service import embedding_service
        logger.info("Scheduling local embedding model pre-warming task in the background...")
        asyncio.create_task(asyncio.to_thread(embedding_service.pre_load_local_model))
    else:
        logger.info("Skipping local embedding pre-warming because EMBEDDING_PROVIDER is 'openai'.")
    
    # Fast fail check for Ollama local server
    if settings.LLM_PROVIDER.lower() == "ollama":
        import httpx
        logger.info(f"Performing fast fail startup check for local Ollama server at '{settings.OLLAMA_API_URL}'...")
        try:
            resp = httpx.get(settings.OLLAMA_API_URL, timeout=3.0)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama server returned status code {resp.status_code}")
            logger.info("Ollama server is verified reachable.")
        except Exception as err:
            critical_msg = (
                f"\n================================================================================\n"
                f"CRITICAL ERROR: Local Ollama server is not reachable at '{settings.OLLAMA_API_URL}'.\n"
                f"Please ensure Ollama is installed and running locally, or switch LLM_PROVIDER config.\n"
                f"Error details: {err}\n"
                f"================================================================================"
            )
            logger.critical(critical_msg)
            import sys
            sys.exit(1)

    logger.info(f"PDF Chatbot backend started in environment: '{settings.ENVIRONMENT}'")
    logger.info(f"CORS origins configured: {settings.cors_origins_list}")

@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("PDF Chatbot backend shutting down...")

