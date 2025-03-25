import os
import json
import glob
import shutil
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
import numpy as np

# Create FastAPI app
app = FastAPI(
    title="Label Studio API",
    description="FastAPI backend for Label Studio-like annotation tool",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ANNOTATION_FOLDER = 'static/annotations'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve React App
app.mount("/", StaticFiles(directory="react-label-studio/build", html=True), name="react")

# Pydantic models for request/response validation
class AnnotationData(BaseModel):
    image: str
    annotations: List[Dict[str, Any]]

class ExportFormat(BaseModel):
    format: str = "json"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API Routes
@app.get("/api/images")
async def get_images():
    """Get list of all uploaded images"""
    image_files = []
    for ext in ALLOWED_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(UPLOAD_FOLDER, f'*.{ext}')))
    
    # Extract just the filenames
    image_files = [os.path.basename(f) for f in image_files]
    
    return {
        'success': True,
        'images': image_files
    }

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """Serve an image file"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.get("/api/annotations/{filename}")
async def get_annotations(filename: str):
    """Get annotations for a specific image"""
    annotation_file = os.path.join(ANNOTATION_FOLDER, f'{os.path.splitext(filename)[0]}.json')
    
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        return {
            'success': True,
            'annotations': annotations
        }
    else:
        return {
            'success': True,
            'annotations': []
        }

@app.post("/api/save-annotation")
async def save_annotation(data: AnnotationData):
    """Save annotations for an image"""
    image_name = data.image
    annotations = data.annotations
    
    # Save to annotation file
    annotation_file = os.path.join(ANNOTATION_FOLDER, f'{os.path.splitext(image_name)[0]}.json')
    
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    return {
        'success': True
    }

@app.post("/api/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """Upload image files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    uploaded_count = 0
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename.replace(" ", "_")  # Simple filename sanitization
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            uploaded_count += 1
            uploaded_files.append(filename)
    
    return {
        'success': True,
        'uploaded': uploaded_count,
        'filenames': uploaded_files
    }

@app.post("/api/upload-annotations")
async def upload_annotations(file: UploadFile = File(...)):
    """Upload annotations file (JSON)"""
    if not file or not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        # Parse the uploaded JSON
        content = await file.read()
        annotations_data = json.loads(content.decode('utf-8'))
        
        # Check if it's a Label Studio format or our simple format
        if isinstance(annotations_data, list) and "predictions" in annotations_data[0]:
            # Label Studio format
            for task in annotations_data:
                image_name = task['data']['image']
                
                # Extract annotations from predictions
                annotations = []
                
                for prediction in task['predictions']:
                    for result in prediction['result']:
                        if result['type'] == 'rectanglelabels':
                            # Convert Label Studio format to our format
                            value = result['value']
                            
                            # Calculate absolute coordinates
                            x = value['x'] * result['original_width'] / 100
                            y = value['y'] * result['original_height'] / 100
                            width = value['width'] * result['original_width'] / 100
                            height = value['height'] * result['original_height'] / 100
                            
                            annotations.append({
                                'id': result['id'],
                                'category': value['rectanglelabels'][0],
                                'bbox': [x, y, width, height],
                                'score': result.get('score', None)
                            })
                
                # Save to annotation file
                annotation_file = os.path.join(ANNOTATION_FOLDER, f'{os.path.splitext(image_name)[0]}.json')
                
                with open(annotation_file, 'w') as f:
                    json.dump(annotations, f, indent=2)
        else:
            # Our simple format - just save as is
            filename = file.filename.replace(" ", "_")
            file_path = os.path.join(ANNOTATION_FOLDER, filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                await file.seek(0)
                buffer.write(await file.read())
        
        return {
            'success': True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete-image/{filename}")
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

@app.post("/api/export")
async def export_annotations(format_data: ExportFormat):
    """Export all annotations in specified format"""
    format_type = format_data.format
    
    # Get all annotation files
    annotation_files = glob.glob(os.path.join(ANNOTATION_FOLDER, '*.json'))
    
    if not annotation_files:
        raise HTTPException(status_code=404, detail="No annotations found")
    
    # Create export directory
    export_dir = os.path.join('static', 'exports')
    os.makedirs(export_dir, exist_ok=True)
    
    if format_type == 'json':
        # Simple JSON export - combine all annotations
        all_annotations = {}
        
        for annotation_file in annotation_files:
            image_name = os.path.basename(annotation_file).replace('.json', '')
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            all_annotations[image_name] = annotations
        
        export_path = os.path.join(export_dir, f'annotations_export_{uuid.uuid4().hex[:8]}.json')
        
        with open(export_path, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        return {
            'success': True,
            'format': 'json',
            'path': export_path
        }
    
    elif format_type == 'coco':
        # COCO format export
        coco_data = {
            "info": {
                "description": "Exported from Simple Label Studio",
                "url": "",
                "version": "1.0",
                "year": 2023,
                "contributor": "Simple Label Studio",
                "date_created": ""
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Collect all categories
        categories = set()
        
        # First pass: collect all categories
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                categories.add(ann['category'])
        
        # Create category list
        for i, category in enumerate(sorted(categories)):
            coco_data['categories'].append({
                "id": i + 1,
                "name": category,
                "supercategory": "none"
            })
        
        # Create category id mapping
        category_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
        
        # Second pass: add images and annotations
        annotation_id = 1
        
        for annotation_file in annotation_files:
            image_name = os.path.basename(annotation_file).replace('.json', '')
            image_path = glob.glob(os.path.join(UPLOAD_FOLDER, f'{image_name}.*'))[0]
            
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Add image
            image_id = len(coco_data['images']) + 1
            coco_data['images'].append({
                "id": image_id,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": ""
            })
            
            # Add annotations
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                x, y, w, h = ann['bbox']
                
                coco_data['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[ann['category']],
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "segmentation": [],
                    "iscrowd": 0
                })
                
                annotation_id += 1
        
        export_path = os.path.join(export_dir, f'annotations_coco_{uuid.uuid4().hex[:8]}.json')
        
        with open(export_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return {
            'success': True,
            'format': 'coco',
            'path': export_path
        }
    
    elif format_type == 'yolo':
        # YOLO format export
        export_path = os.path.join(export_dir, f'yolo_export_{uuid.uuid4().hex[:8]}')
        os.makedirs(export_path, exist_ok=True)
        
        # Create images directory
        images_dir = os.path.join(export_path, 'images')
        labels_dir = os.path.join(export_path, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Collect all categories
        categories = set()
        
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                categories.add(ann['category'])
        
        # Create category mapping
        category_list = sorted(list(categories))
        category_map = {cat: i for i, cat in enumerate(category_list)}
        
        # Save category mapping
        with open(os.path.join(export_path, 'classes.txt'), 'w') as f:
            for cat in category_list:
                f.write(f"{cat}\n")
        
        # Process each annotation file
        for annotation_file in annotation_files:
            image_name = os.path.basename(annotation_file).replace('.json', '')
            image_path = glob.glob(os.path.join(UPLOAD_FOLDER, f'{image_name}.*'))[0]
            
            # Copy image to export directory
            shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
            
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            # Create YOLO format annotation file
            yolo_file = os.path.join(labels_dir, f"{image_name}.txt")
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            with open(yolo_file, 'w') as f:
                for ann in annotations:
                    category_id = category_map[ann['category']]
                    x, y, width, height = ann['bbox']
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    center_x = (x + width / 2) / img_width
                    center_y = (y + height / 2) / img_height
                    norm_width = width / img_width
                    norm_height = height / img_height
                    
                    f.write(f"{category_id} {center_x} {center_y} {norm_width} {norm_height}\n")
        
        return {
            'success': True,
            'format': 'yolo',
            'path': export_path
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format_type}")

# Add a route to generate Label Studio predictions
@app.post("/api/generate-predictions")
async def generate_predictions(
    frames_dir: str = Form(...),
    output_file: str = Form(...),
    model_name: str = Form(...),
    text_prompt: str = Form("person, car, bicycle, motorcycle, truck, traffic light"),
    confidence_threshold: float = Form(0.25),
    include_masks: bool = Form(False)
):
    """
    Generate pre-annotations in Label Studio format from object detection results.
    
    This endpoint calls the generate_label_studio_predictions function from extractframefromvideo_seg.py
    """
    try:
        # Import the function from the other module
        from extractframefromvideo_seg import generate_label_studio_predictions
        
        # Call the function
        predictions = generate_label_studio_predictions(
            frames_dir=frames_dir,
            output_file=output_file,
            model_name=model_name,
            text_prompt=text_prompt,
            confidence_threshold=confidence_threshold,
            include_masks=include_masks
        )
        
        return {
            'success': True,
            'message': f"Generated predictions for {len(predictions)} images",
            'output_file': output_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("fastapi_label_studio:app", host="0.0.0.0", port=8000, reload=True)