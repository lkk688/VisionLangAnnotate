from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import glob
from ..config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from ..utils import allowed_file

router = APIRouter(prefix="/images", tags=["Images"])

@router.get("/")
async def get_images():
    """Get list of all uploaded images"""
    image_files = []
    for ext in ALLOWED_EXTENSIONS:
        #image_files.extend(glob.glob(str(UPLOAD_FOLDER / f'*.{ext}')))
        image_files.extend(glob.glob(os.path.join(UPLOAD_FOLDER, f'*.{ext}')))
    
    # Extract just the filenames
    image_files = [os.path.basename(f) for f in image_files]
    
    return {
        'success': True,
        'images': image_files
    }

@router.get("/{filename}")
async def get_image(filename: str):
    """Serve an image file"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@router.delete("/delete-image/{filename}")
async def delete_image(filename: str):
    """Delete an image and its annotations"""
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    annotation_path = os.path.join(ANNOTATION_FOLDER, f'{os.path.splitext(filename)[0]}.json')
    
    # Delete image if it exists
    if os.path.exists(image_path):
        os.remove(image_path)
    
    # Delete annotation if it exists
    if os.path.exists(annotation_path):
        os.remove(annotation_path)
    
    return {
        'success': True
    }