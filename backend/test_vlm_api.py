#!/usr/bin/env python3
"""
Test script to verify VLM backend API endpoints are properly configured.
This script doesn't actually run the endpoints (which require the pipeline to be initialized),
but verifies that the FastAPI app is configured correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/lkk/Developer/VisionLangAnnotate')

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from backend.src.models import (
            ImageDescriptionRequest,
            ImageDescriptionResponse,
            ObjectDetectionRequest,
            ObjectDetectionResponse,
            VideoAnalysisRequest,
            VideoAnalysisResponse,
            BackendInfoResponse
        )
        print("✓ Models imported successfully")
    except Exception as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    try:
        from backend.src.config import (
            VLM_MODEL_NAME,
            VLM_BACKEND,
            VLM_DEVICE,
            VLM_OUTPUT_DIR
        )
        print(f"✓ Config loaded successfully")
        print(f"  - Model: {VLM_MODEL_NAME}")
        print(f"  - Backend: {VLM_BACKEND}")
        print(f"  - Device: {VLM_DEVICE}")
        print(f"  - Output Dir: {VLM_OUTPUT_DIR}")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from backend.src.api import vlm
        print("✓ VLM router imported successfully")
    except Exception as e:
        print(f"✗ Failed to import VLM router: {e}")
        return False
    
    try:
        from backend.src.main import app
        print("✓ FastAPI app imported successfully")
    except Exception as e:
        print(f"✗ Failed to import FastAPI app: {e}")
        return False
    
    return True


def test_routes():
    """Test that VLM routes are registered"""
    print("\nTesting routes...")
    
    try:
        from backend.src.main import app
        
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        vlm_routes = [r for r in routes if 'vlm' in r]
        
        print(f"✓ Total routes: {len(routes)}")
        print(f"✓ VLM routes: {len(vlm_routes)}")
        
        expected_routes = [
            '/api/vlm/backend-info',
            '/api/vlm/describe-image/{filename}',
            '/api/vlm/detect-objects/{filename}',
            '/api/vlm/analyze-video/{filename}',
            '/api/vlm/visualization/{filename}',
            '/api/vlm/segmentation/{filename}',
            '/api/vlm/annotation/{filename}'
        ]
        
        print("\nExpected VLM routes:")
        for route in expected_routes:
            if route in vlm_routes:
                print(f"  ✓ {route}")
            else:
                print(f"  ✗ {route} (missing)")
        
        return len(vlm_routes) == len(expected_routes)
    
    except Exception as e:
        print(f"✗ Failed to test routes: {e}")
        return False


def test_openapi_schema():
    """Test that OpenAPI schema is generated correctly"""
    print("\nTesting OpenAPI schema...")
    
    try:
        from backend.src.main import app
        
        schema = app.openapi()
        
        # Check if VLM endpoints are in the schema
        paths = schema.get('paths', {})
        vlm_paths = [p for p in paths.keys() if 'vlm' in p]
        
        print(f"✓ OpenAPI schema generated")
        print(f"✓ VLM endpoints in schema: {len(vlm_paths)}")
        
        # Check for specific models in components
        components = schema.get('components', {}).get('schemas', {})
        vlm_models = [
            'ImageDescriptionRequest',
            'ImageDescriptionResponse',
            'ObjectDetectionRequest',
            'ObjectDetectionResponse',
            'VideoAnalysisRequest',
            'VideoAnalysisResponse',
            'BackendInfoResponse'
        ]
        
        print("\nVLM models in schema:")
        for model in vlm_models:
            if model in components:
                print(f"  ✓ {model}")
            else:
                print(f"  ✗ {model} (missing)")
        
        return True
    
    except Exception as e:
        print(f"✗ Failed to test OpenAPI schema: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("VLM Backend API Verification")
    print("="*60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_routes():
        all_passed = False
    
    if not test_openapi_schema():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now:")
        print("1. Start the backend server: uvicorn backend.src.main:app --reload")
        print("2. View API docs at: http://localhost:8000/docs")
        print("3. Test VLM endpoints at: http://localhost:8000/api/vlm/")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
    print("="*60)


if __name__ == '__main__':
    main()
