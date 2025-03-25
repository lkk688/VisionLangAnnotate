from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import glob
from ..config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from ..utils import allowed_file

router = APIRouter(prefix="/images", tags=["Images"])

@router.get("/")
async def get_images():
    image_files = []
    for ext in ALLOWED_EXTENSIONS:
        image_files.extend(glob.glob(str(UPLOAD_FOLDER / f'*.{ext}')))
    return {
        'success': True,
        'images': [Path(f).name for f in image_files]
    }

@router.get("/{filename}")
async def get_image(filename: str):
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)