from pydantic import BaseModel
from typing import List, Dict, Any

class AnnotationData(BaseModel):
    image: str
    annotations: List[Dict[str, Any]]

class ExportFormat(BaseModel):
    format: str = "json"