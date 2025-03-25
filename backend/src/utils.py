from PIL import Image
from pathlib import Path
from .config import ALLOWED_EXTENSIONS

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_dimensions(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size