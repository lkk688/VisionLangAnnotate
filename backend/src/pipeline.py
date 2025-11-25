"""
Pipeline initialization and management for QwenObjectDetectionPipeline.
Provides a singleton pattern for the VLM pipeline to avoid re-initializing the model.
"""

import os
import sys
from typing import Optional
from pathlib import Path

# Add VLM directory to path
vlm_path = Path(__file__).parent.parent.parent / 'VisionLangAnnotateModels' / 'VLM'
if str(vlm_path) not in sys.path:
    sys.path.insert(0, str(vlm_path))

# Import the pipeline
try:
    from qwen_object_detection_pipeline3 import QwenObjectDetectionPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"Warning: QwenObjectDetectionPipeline not available: {e}")
    QwenObjectDetectionPipeline = None

from .config import (
    VLM_MODEL_NAME,
    VLM_BACKEND,
    VLM_DEVICE,
    VLM_OUTPUT_DIR,
    ENABLE_SAM,
    ENABLE_TRADITIONAL_DETECTORS,
    TRADITIONAL_DETECTORS
)

# Global pipeline instance
_pipeline_instance: Optional[QwenObjectDetectionPipeline] = None


def get_pipeline() -> Optional[QwenObjectDetectionPipeline]:
    """
    Get or create the QwenObjectDetectionPipeline singleton instance.
    
    Returns:
        QwenObjectDetectionPipeline instance or None if initialization fails
    """
    global _pipeline_instance
    
    if _pipeline_instance is not None:
        return _pipeline_instance
    
    if not PIPELINE_AVAILABLE:
        print("Error: QwenObjectDetectionPipeline not available")
        return None
    
    try:
        print(f"Initializing QwenObjectDetectionPipeline...")
        print(f"  Model: {VLM_MODEL_NAME}")
        print(f"  Backend: {VLM_BACKEND}")
        print(f"  Device: {VLM_DEVICE}")
        print(f"  Output Dir: {VLM_OUTPUT_DIR}")
        print(f"  SAM Enabled: {ENABLE_SAM}")
        print(f"  Traditional Detectors Enabled: {ENABLE_TRADITIONAL_DETECTORS}")
        
        _pipeline_instance = QwenObjectDetectionPipeline(
            model_name=VLM_MODEL_NAME,
            device=VLM_DEVICE,
            output_dir=str(VLM_OUTPUT_DIR),
            enable_sam=ENABLE_SAM,
            enable_traditional_detectors=ENABLE_TRADITIONAL_DETECTORS,
            traditional_detectors=TRADITIONAL_DETECTORS if ENABLE_TRADITIONAL_DETECTORS else None,
            vlm_backend=VLM_BACKEND
        )
        
        print("Pipeline initialized successfully")
        return _pipeline_instance
        
    except Exception as e:
        print(f"Error: Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


def reset_pipeline():
    """Reset the pipeline instance (useful for testing or configuration changes)"""
    global _pipeline_instance
    _pipeline_instance = None
    print("Pipeline instance reset")
