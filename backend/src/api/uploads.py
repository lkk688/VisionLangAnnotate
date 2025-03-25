from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from ..config import UPLOAD_FOLDER, ANNOTATION_FOLDER
from ..utils import allowed_file

router = APIRouter(prefix="/upload", tags=["Uploads"])

@router.post("/")
async def upload_file(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename.replace(" ", "_")
            file_path = UPLOAD_FOLDER / filename
            file_path.write_bytes(await file.read())
            uploaded_files.append(filename)
    
    return {
        'success': True,
        'uploaded': len(uploaded_files),
        'filenames': uploaded_files
    }

@router.delete("/{filename}")
async def delete_image(filename: str):
    image_path = UPLOAD_FOLDER / filename
    annotation_path = ANNOTATION_FOLDER / f'{Path(filename).stem}.json'
    
    if image_path.exists():
        image_path.unlink()
    if annotation_path.exists():
        annotation_path.unlink()
    
    return {'success': True}