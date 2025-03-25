import os
from pathlib import Path

# #Configuration
# # Method A: Using project root detection
# PROJECT_ROOT = Path(__file__).parent.parent.parent  # Adjust based on your structure
# UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
# # Method B: If running from project root
# #UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ANNOTATION_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads") #'static/annotations'

# MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

# Create directories if they don't exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

BASE_DIR = Path(__file__).parent.parent
UPLOAD_FOLDER = BASE_DIR.parent / "static" / "uploads"
ANNOTATION_FOLDER = BASE_DIR.parent / "static" / "annotations"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
print("UPLOAD_FOLDER:", UPLOAD_FOLDER)
# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ANNOTATION_FOLDER.mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS