from fastapi import APIRouter, HTTPException
import json
from pathlib import Path
import uuid
from ..config import ANNOTATION_FOLDER
from ..models import AnnotationData, ExportFormat
import os

router = APIRouter(prefix="/annotations", tags=["Annotations"])

#"/api/annotations/{filename}"
@router.get("/{filename}")
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

@router.post("/save")
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

@router.post("/export")
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