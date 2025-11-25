from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from .api import health, images, annotations, uploads, vlm


# Create FastAPI app
app = FastAPI(
    title="Annotation Tool API",
    description="Backend for annotation tool",
    version="1.0.0"
)

# Static files
static_path = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Allow CORS for React frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(images.router, prefix="/api")
app.include_router(annotations.router, prefix="/api")
app.include_router(uploads.router, prefix="/api")
app.include_router(vlm.router, prefix="/api")


    
@app.get("/")
def read_root():
    return {"message": "Annotation Tool Backend"}

# if __name__ == "__main__":
#     uvicorn.run("fastapi_backend:app", host="0.0.0.0", port=8000, reload=True)