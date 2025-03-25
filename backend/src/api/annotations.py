from fastapi import APIRouter, HTTPException
import json
from pathlib import Path
import uuid
from ..config import ANNOTATION_FOLDER
from ..models import AnnotationData, ExportFormat

router = APIRouter(prefix="/annotations", tags=["Annotations"])

@router.get("/{filename}")
async def get_annotations(filename: str):
    annotation_file = ANNOTATION_FOLDER / f'{Path(filename).stem}.json'
    if annotation_file.exists():
        return {
            'success': True,
            'annotations': json.loads(annotation_file.read_text())
        }
    return {'success': True, 'annotations': []}

@router.post("/save")
async def save_annotation(data: AnnotationData):
    annotation_file = ANNOTATION_FOLDER / f'{Path(data.image).stem}.json'
    annotation_file.write_text(json.dumps(data.annotations, indent=2))
    return {'success': True}

@router.post("/export")
async def export_annotations(format_data: ExportFormat):
    annotation_files = list(ANNOTATION_FOLDER.glob('*.json'))
    if not annotation_files:
        raise HTTPException(status_code=404, detail="No annotations found")
    
    export_dir = ANNOTATION_FOLDER.parent / "exports"
    export_dir.mkdir(exist_ok=True)
    
    if format_data.format == 'json':
        export_path = export_dir / f'export_{uuid.uuid4().hex[:8]}.json'
        all_annotations = {
            f.stem: json.loads(f.read_text())
            for f in annotation_files
        }
        export_path.write_text(json.dumps(all_annotations, indent=2))
        return {
            'success': True,
            'format': 'json',
            'path': str(export_path.relative_to(ANNOTATION_FOLDER.parent.parent))
        }
    
    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format_data.format}")