from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Existing models
class AnnotationData(BaseModel):
    image: str
    annotations: List[Dict[str, Any]]

class ExportFormat(BaseModel):
    format: str = "json"

# VLM API Models
class ImageDescriptionRequest(BaseModel):
    custom_prompt: Optional[str] = Field(None, description="Optional custom prompt for image description")

class ImageDescriptionResponse(BaseModel):
    description: str
    success: bool = True

class DetectedObject(BaseModel):
    label: str
    bbox: List[float] = Field(..., description="Bounding box in [x1, y1, x2, y2] format")
    description: Optional[str] = None
    confidence: float
    source: Optional[str] = None

class ObjectDetectionRequest(BaseModel):
    detection_method: str = Field("Hybrid Mode", description="Detection method: 'VLM Only', 'Hybrid Mode', or 'Hybrid-Sequential'")
    use_sam_segmentation: bool = Field(True, description="Enable SAM segmentation")

class ObjectDetectionResponse(BaseModel):
    objects: List[DetectedObject]
    raw_response: str
    visualization_paths: List[str] = Field(default_factory=list)
    segmentation_paths: List[str] = Field(default_factory=list)
    json_path: Optional[str] = None
    detection_method: str
    num_objects: int
    success: bool = True

class VideoAnalysisRequest(BaseModel):
    task_type: str = Field("detection", description="Analysis type: 'description' or 'detection'")
    apply_privacy: bool = Field(False, description="Apply privacy filtering")
    use_sam_segmentation: bool = Field(False, description="Enable SAM segmentation")

class FrameResult(BaseModel):
    frame_number: int
    objects: List[DetectedObject] = Field(default_factory=list)
    description: Optional[str] = None
    visualization_path: Optional[str] = None
    json_path: Optional[str] = None

class VideoAnalysisResponse(BaseModel):
    frames: List[FrameResult]
    total_frames: int
    task_type: str
    success: bool = True
    message: Optional[str] = None

class BackendInfoResponse(BaseModel):
    backend_type: str
    model_name: str
    device: str
    status: str
    additional_info: Optional[Dict[str, Any]] = None