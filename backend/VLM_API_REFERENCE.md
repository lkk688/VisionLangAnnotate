# VLM Backend API - Quick Reference

## Starting the Server

```bash
cd /home/lkk/Developer/VisionLangAnnotate
uvicorn backend.src.main:app --reload --host 0.0.0.0 --port 8000
```

## API Base URL

```
http://localhost:8000/api/vlm
```

## Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### 1. Get Backend Info
```bash
GET /api/vlm/backend-info
```

### 2. Describe Image
```bash
POST /api/vlm/describe-image/{filename}
Content-Type: application/json

{
  "custom_prompt": "Describe this image" (optional)
}
```

### 3. Detect Objects
```bash
POST /api/vlm/detect-objects/{filename}
Content-Type: application/json

{
  "detection_method": "Hybrid Mode",  # VLM Only | Hybrid Mode | Hybrid-Sequential
  "use_sam_segmentation": true
}
```

### 4. Analyze Video
```bash
POST /api/vlm/analyze-video/{filename}
Content-Type: application/json

{
  "task_type": "detection",  # detection | description
  "apply_privacy": false,
  "use_sam_segmentation": false
}
```

### 5. Get Visualization
```bash
GET /api/vlm/visualization/{filename}
```

### 6. Get Segmentation
```bash
GET /api/vlm/segmentation/{filename}
```

### 7. Get Annotation
```bash
GET /api/vlm/annotation/{filename}
```

## Example Usage

### Python

```python
import requests

# Describe an image
response = requests.post(
    "http://localhost:8000/api/vlm/describe-image/test.jpg",
    json={"custom_prompt": "What do you see?"}
)
print(response.json())

# Detect objects
response = requests.post(
    "http://localhost:8000/api/vlm/detect-objects/test.jpg",
    json={
        "detection_method": "Hybrid Mode",
        "use_sam_segmentation": True
    }
)
results = response.json()
print(f"Found {results['num_objects']} objects")
```

### JavaScript/Fetch

```javascript
// Describe image
const describeImage = async (filename) => {
  const response = await fetch(`/api/vlm/describe-image/${filename}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({custom_prompt: 'Describe this image'})
  });
  return await response.json();
};

// Detect objects
const detectObjects = async (filename) => {
  const response = await fetch(`/api/vlm/detect-objects/${filename}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      detection_method: 'Hybrid Mode',
      use_sam_segmentation: true
    })
  });
  return await response.json();
};
```

### cURL

```bash
# Describe image
curl -X POST "http://localhost:8000/api/vlm/describe-image/test.jpg" \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "What is in this image?"}'

# Detect objects
curl -X POST "http://localhost:8000/api/vlm/detect-objects/test.jpg" \
  -H "Content-Type: application/json" \
  -d '{
    "detection_method": "Hybrid Mode",
    "use_sam_segmentation": true
  }'

# Get backend info
curl -X GET "http://localhost:8000/api/vlm/backend-info"
```

## Configuration (Environment Variables)

```bash
# Model configuration
export VLM_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export VLM_BACKEND="huggingface"  # huggingface | vllm | ollama
export VLM_DEVICE="cuda"           # cuda | cpu

# Output directory
export VLM_OUTPUT_DIR="./output/detection_results"

# Feature flags
export ENABLE_SAM="true"
export ENABLE_TRADITIONAL_DETECTORS="true"
export TRADITIONAL_DETECTORS="yolo,detr"
```

## Response Formats

### Image Description Response
```json
{
  "description": "A red car parked on the street...",
  "success": true
}
```

### Object Detection Response
```json
{
  "objects": [
    {
      "label": "Car",
      "bbox": [100, 50, 200, 150],
      "description": "red sedan parked",
      "confidence": 0.95,
      "source": "hybrid_vlm"
    }
  ],
  "raw_response": "...",
  "visualization_paths": ["viz_001.jpg"],
  "segmentation_paths": ["seg_001.jpg"],
  "json_path": "ann_001.json",
  "detection_method": "Hybrid Mode",
  "num_objects": 1,
  "success": true
}
```

### Video Analysis Response
```json
{
  "frames": [
    {
      "frame_number": 1,
      "objects": [...],
      "description": "...",
      "visualization_path": "frame_001.jpg",
      "json_path": "frame_001.json"
    }
  ],
  "total_frames": 10,
  "task_type": "detection",
  "success": true
}
```

## Error Codes

- `200` - Success
- `404` - File not found
- `500` - Processing error
- `503` - Pipeline not available

## Notes

- Images must be uploaded to `/api/upload` first
- Files are referenced by filename only
- Output files are automatically cleaned before each detection
- Pipeline is initialized once and reused (singleton pattern)
