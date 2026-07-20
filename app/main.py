from fastapi import FastAPI
from app.api.v1.documents import router as document_router
from app.config.settings import settings
from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.upload import router as upload_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

# allow_origins=["*"] + allow_credentials=False is the correct setting for a
# local dev project with no session cookies or Authorization headers.
# NOTE: allow_credentials=True + "null" origin (file://) causes Starlette to
# return 400 on OPTIONS preflight because the spec forbids credentialed
# wildcard-like responses to opaque origins. Our frontend uses plain fetch()
# with no `credentials: 'include'`, so credentials mode is irrelevant here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(document_router)
@app.get("/")
def root():

    return {
        "message": "Welcome to ForgeMind AI 🚀"
    }