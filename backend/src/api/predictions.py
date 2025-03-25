from fastapi import APIRouter, HTTPException
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
import json
from pathlib import Path
import uuid
from ..config import ANNOTATION_FOLDER
from ..models import AnnotationData, ExportFormat
import os

router = APIRouter(prefix="/predictions", tags=["Predictions"])


# Add a route to generate Label Studio predictions
@app.post("/generate-predictions")
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
        from ..extractframefromvideo import generate_label_studio_predictions
        
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