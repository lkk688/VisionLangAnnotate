"""
VLM (Vision Language Model) API endpoints for image description, 
object detection, and video analysis.
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, File, UploadFile, Body
from fastapi.responses import FileResponse

from ..config import UPLOAD_FOLDER, VLM_OUTPUT_DIR
from ..models import (
    ImageDescriptionRequest,
    ImageDescriptionResponse,
    ObjectDetectionRequest,
    ObjectDetectionResponse,
    DetectedObject,
    VideoAnalysisRequest,
    VideoAnalysisResponse,
    FrameResult,
    BackendInfoResponse
)
from ..pipeline import get_pipeline

router = APIRouter(prefix="/vlm", tags=["VLM"])


def _clear_detection_output():
    """Clear the detection output directory before each detection call"""
    try:
        if VLM_OUTPUT_DIR.exists():
            # Clear all subdirectories
            for subdir in ['json_annotations', 'raw_responses', 'segmentations', 'visualizations']:
                subdir_path = VLM_OUTPUT_DIR / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
                    subdir_path.mkdir(exist_ok=True)
                    print(f"Cleared {subdir_path}")
    except Exception as e:
        print(f"Warning: Could not clear detection output directory: {e}")


def _get_saved_files(pattern: str) -> List[str]:
    """Get saved files matching the pattern, sorted by modification time"""
    files = glob.glob(pattern)
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
    return files


@router.get("/backend-info", response_model=BackendInfoResponse)
async def get_backend_info():
    """Get information about the current VLM backend status"""
    pipeline = get_pipeline()
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="VLM pipeline not available")
    
    # Get backend info from pipeline
    if hasattr(pipeline, 'vlm_backend_info'):
        backend_info = pipeline.vlm_backend_info
    elif hasattr(pipeline, '_get_vlm_backend_info'):
        backend_info = pipeline._get_vlm_backend_info()
    else:
        backend_info = {
            'backend_type': 'Unknown',
            'model_name': pipeline.model_name if hasattr(pipeline, 'model_name') else 'Unknown',
            'device': pipeline.device if hasattr(pipeline, 'device') else 'Unknown',
            'status': 'active'
        }
    
    return BackendInfoResponse(
        backend_type=backend_info.get('backend_type', 'Unknown'),
        model_name=backend_info.get('model_name', 'Unknown'),
        device=backend_info.get('device', 'Unknown'),
        status=backend_info.get('status', 'active'),
        additional_info=backend_info
    )


@router.post("/describe-image/{filename}", response_model=ImageDescriptionResponse)
async def describe_image(
    filename: str,
    request: ImageDescriptionRequest = Body(default=ImageDescriptionRequest())
):
    """
    Generate a description for an uploaded image.
    
    Args:
        filename: Name of the image file in the uploads directory
        request: Optional custom prompt for description
    
    Returns:
        Image description text
    """
    pipeline = get_pipeline()
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="VLM pipeline not available")
    
    # Construct image path
    image_path = UPLOAD_FOLDER / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    try:
        # Call describe_image with optional custom prompt
        if request.custom_prompt:
            description = pipeline.describe_image(str(image_path), request.custom_prompt)
        else:
            description = pipeline.describe_image(str(image_path))
        
        return ImageDescriptionResponse(
            description=description,
            success=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating description: {str(e)}")


@router.post("/detect-objects/{filename}", response_model=ObjectDetectionResponse)
async def detect_objects(
    filename: str,
    request: ObjectDetectionRequest = Body(default=ObjectDetectionRequest())
):
    """
    Detect objects in an uploaded image.
    
    Args:
        filename: Name of the image file in the uploads directory
        request: Detection parameters (method, SAM segmentation)
    
    Returns:
        Detection results with objects, visualizations, and JSON annotations
    """
    pipeline = get_pipeline()
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="VLM pipeline not available")
    
    # Construct image path
    image_path = UPLOAD_FOLDER / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    # Clear output directory before detection
    _clear_detection_output()
    
    try:
        # Choose detection method
        if request.detection_method == "VLM Only":
            # VLM-only detection
            result = pipeline.detect_objects(
                str(image_path),
                use_sam_segmentation=request.use_sam_segmentation
            )
        elif request.detection_method == "Hybrid-Sequential":
            # Hybrid-Sequential detection
            result = pipeline.detect_objects_hybrid(
                str(image_path),
                use_sam_segmentation=request.use_sam_segmentation,
                sequential_mode=True,
                cropped_sequential_mode=False,
                save_results=True
            )
        else:  # Hybrid Mode (default)
            # Hybrid detection
            result = pipeline.detect_objects_hybrid(
                str(image_path),
                use_sam_segmentation=request.use_sam_segmentation,
                sequential_mode=False,
                cropped_sequential_mode=False,
                save_results=True
            )
        
        # Extract results
        objects = result.get('objects', [])
        raw_response = result.get('raw_response', '')
        
        # Convert objects to DetectedObject models
        detected_objects = []
        for obj in objects:
            detected_objects.append(DetectedObject(
                label=obj.get('label', 'Unknown'),
                bbox=obj.get('bbox', [0, 0, 0, 0]),
                description=obj.get('description', ''),
                confidence=obj.get('confidence', 0.0),
                source=obj.get('source', None)
            ))
        
        # Get saved visualization and segmentation images
        visualization_paths = []
        segmentation_paths = []
        
        if hasattr(pipeline, 'output_dir') and pipeline.output_dir:
            # Look for visualization images
            viz_pattern = str(Path(pipeline.output_dir) / "visualizations" / "*")
            viz_files = _get_saved_files(viz_pattern)
            visualization_paths = [os.path.basename(f) for f in viz_files[:5]]
            
            # Look for segmentation images
            seg_pattern = str(Path(pipeline.output_dir) / "segmentations" / "*")
            seg_files = _get_saved_files(seg_pattern)
            segmentation_paths = [os.path.basename(f) for f in seg_files[:5]]
        
        # Find JSON annotation file
        json_path = None
        if hasattr(pipeline, 'output_dir') and pipeline.output_dir:
            json_pattern = str(Path(pipeline.output_dir) / "json_annotations" / "*.json")
            json_files = _get_saved_files(json_pattern)
            if json_files:
                json_path = os.path.basename(json_files[0])
        
        return ObjectDetectionResponse(
            objects=detected_objects,
            raw_response=raw_response,
            visualization_paths=visualization_paths,
            segmentation_paths=segmentation_paths,
            json_path=json_path,
            detection_method=request.detection_method,
            num_objects=len(detected_objects),
            success=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in object detection: {str(e)}")


@router.post("/analyze-video/{filename}", response_model=VideoAnalysisResponse)
async def analyze_video(
    filename: str,
    request: VideoAnalysisRequest = Body(default=VideoAnalysisRequest())
):
    """
    Analyze a video file (either description or detection).
    
    Args:
        filename: Name of the video file in the uploads directory
        request: Analysis parameters (task type, privacy, SAM segmentation)
    
    Returns:
        Frame-by-frame analysis results
    """
    pipeline = get_pipeline()
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="VLM pipeline not available")
    
    # Construct video path
    video_path = UPLOAD_FOLDER / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {filename}")
    
    try:
        # Call pipeline for video analysis
        result = pipeline.detect_objects(
            image_path=str(video_path),
            save_results=True,
            apply_privacy=request.apply_privacy,
            use_sam_segmentation=request.use_sam_segmentation
        )
        
        frames = []
        
        if isinstance(result, list):
            # Multiple frames processed
            for i, frame_result in enumerate(result):
                objects = frame_result.get('objects', [])
                detected_objects = []
                
                for obj in objects:
                    detected_objects.append(DetectedObject(
                        label=obj.get('label', 'Unknown'),
                        bbox=obj.get('bbox', [0, 0, 0, 0]),
                        description=obj.get('description', ''),
                        confidence=obj.get('confidence', 0.0),
                        source=obj.get('source', None)
                    ))
                
                frame_info = FrameResult(
                    frame_number=i + 1,
                    objects=detected_objects,
                    description=frame_result.get('raw_response', None) if request.task_type == "description" else None,
                    visualization_path=frame_result.get('visualization_path', None),
                    json_path=frame_result.get('json_path', None)
                )
                frames.append(frame_info)
        else:
            # Single frame result
            objects = result.get('objects', [])
            detected_objects = []
            
            for obj in objects:
                detected_objects.append(DetectedObject(
                    label=obj.get('label', 'Unknown'),
                    bbox=obj.get('bbox', [0, 0, 0, 0]),
                    description=obj.get('description', ''),
                    confidence=obj.get('confidence', 0.0),
                    source=obj.get('source', None)
                ))
            
            frame_info = FrameResult(
                frame_number=1,
                objects=detected_objects,
                description=result.get('raw_response', None) if request.task_type == "description" else None,
                visualization_path=result.get('visualization_path', None),
                json_path=result.get('json_path', None)
            )
            frames.append(frame_info)
        
        return VideoAnalysisResponse(
            frames=frames,
            total_frames=len(frames),
            task_type=request.task_type,
            success=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in video analysis: {str(e)}")


@router.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Get a visualization image file"""
    file_path = VLM_OUTPUT_DIR / "visualizations" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(file_path)


@router.get("/segmentation/{filename}")
async def get_segmentation(filename: str):
    """Get a segmentation image file"""
    file_path = VLM_OUTPUT_DIR / "segmentations" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not found")
    
    return FileResponse(file_path)


@router.get("/annotation/{filename}")
async def get_annotation(filename: str):
    """Get a JSON annotation file"""
    file_path = VLM_OUTPUT_DIR / "json_annotations" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    return FileResponse(file_path)
