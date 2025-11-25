from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import json
from ..config import UPLOAD_FOLDER, ANNOTATION_FOLDER
from ..utils import allowed_file

router = APIRouter(prefix="/upload", tags=["Uploads"])

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single image file"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Simple filename sanitization
    filename = file.filename.replace(" ", "_")
    file_path = UPLOAD_FOLDER / filename
    
    # Save the file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    return {
        'success': True,
        'filename': filename,
        'uploaded': 1
    }

@router.post("/upload-annotation")
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
    
@router.delete("/{filename}")
async def delete_image(filename: str):
    image_path = UPLOAD_FOLDER / filename
    annotation_path = ANNOTATION_FOLDER / f'{Path(filename).stem}.json'
    
    if image_path.exists():
        image_path.unlink()
    if annotation_path.exists():
        annotation_path.unlink()
    
    return {'success': True}