# Multi-Step Pipeline for Object Detection and Classification

This pipeline combines object detection results from Label Studio annotations with detailed classification using Vision-Language Models (VLMs).

## Overview

The pipeline consists of two or three main steps:

1. **First Step**: Parse Label Studio annotation JSON files to extract object bounding boxes
2. **Second Step**: Use the `HuggingFaceVLM` class from `vlm_classifierv3.py` to generate detailed descriptions for each detected object
3. **Third Step (Optional)**: Use a local Ollama LLM model to standardize VLM outputs into predefined classes. This step processes all objects from a single image together in batch mode for improved efficiency.

## Requirements

- Python 3.8+
- PIL (Pillow)
- PyTorch
- Transformers
- Label Studio annotation JSON files

## Usage

```bash
python two_step_pipeline.py \
    --image_dir /path/to/images \
    --annotation_file /path/to/labelstudio_annotations.json \
    --model_name Salesforce/blip2-opt-2.7b \
    --device cuda \
    --output_file pipeline_results.json \
    --ollama_model llama2 \
    --allowed_classes "Car,Truck,Pedestrian,Bicycle,Traffic light"
```

### Arguments

- `--image_dir`: Directory containing the images referenced in the Label Studio annotations
- `--annotation_file`: Path to the Label Studio annotations JSON file
- `--model_name`: Name of the HuggingFaceVLM model to use (default: "Salesforce/blip2-opt-2.7b")
- `--device`: Device to run the model on ("cuda" or "cpu", default: "cuda")
- `--output_file`: Path to save the results (default: "pipeline_results.json")
- `--ollama_model`: (Optional) Name of the Ollama model to use for post-processing (e.g., "llama2", "mistral", etc.)
- `--allowed_classes`: (Optional) Comma-separated list of allowed class names for standardization

## Supported VLM Models

The pipeline supports various Vision-Language Models through the `HuggingFaceVLM` class:

- **BLIP2**: Salesforce/blip2-opt-2.7b, Salesforce/blip2-flan-t5-xxl, etc.
- **LLaVA**: llava-hf/llava-1.5-7b-hf, llava-hf/llava-v1.6-mistral-7b-hf, etc.
- **SmolVLM**: HuggingFaceTB/SmolVLM-Instruct, HuggingFaceTB/SmolVLM-Base, etc.

## Batch Processing with Ollama

When using the optional third step with Ollama, the pipeline processes all objects from a single image together in a single API call, rather than processing each object individually. This approach offers several benefits:

1. **Improved Efficiency**: Reduces the number of API calls to the Ollama service
2. **Reduced Processing Time**: Significantly decreases the overall processing time for images with multiple objects
3. **Contextual Understanding**: Allows the LLM to consider all objects in an image together, potentially improving classification accuracy

The batch processing automatically groups objects by their source image and sends them to Ollama as a batch, then maps the results back to the individual objects.
- **MiniGPT-4**: Vision-CAIR/MiniGPT-4
- **GLIP**: GLIPModel/GLIP (for object detection and grounding)

## Examples

### Two-Step Pipeline Example

```bash
python two_step_pipeline.py \
    --image_dir /home/lkk/Developer/VisionLangAnnotate/output/dashcam_videos/onevideo_yolo11l_v2/20250613_061117_NF_20250710_135215 \
    --annotation_file /home/lkk/Developer/VisionLangAnnotate/output/dashcam_videos/onevideo_yolo11l_v2/20250613_061117_NF_20250710_135215/labelstudio_annotations.json \
    --model_name Salesforce/blip2-opt-2.7b \
    --output_file two_step_results.json
```

### Three-Step Pipeline Example with Ollama

```bash
python two_step_pipeline.py \
    --image_dir /home/lkk/Developer/VisionLangAnnotate/output/dashcam_videos/onevideo_yolo11l_v2/20250613_061117_NF_20250710_135215 \
    --annotation_file /home/lkk/Developer/VisionLangAnnotate/output/dashcam_videos/onevideo_yolo11l_v2/20250613_061117_NF_20250710_135215/labelstudio_annotations.json \
    --model_name Salesforce/blip2-opt-2.7b \
    --output_file three_step_results.json \
    --ollama_model llama2 \
    --allowed_classes "Car,Truck,Vehicle blocking bike lane,Burned vehicle,Police car,Pedestrian,Worker,Street vendor,Residential trash bin,Commercial dumpster,Street sign,Construction sign,Traffic signal light,Broken traffic lights,Tree,Overhanging branch,Dumped trash,Yard waste,Glass/debris,Pothole,Unclear bike lane markings,Utility pole,Downed bollard,Cone,Streetlight outage,Graffiti,Bench,Vehicle in bike lane,Bicycle,Scooter,Wheelchair,Bus,Train,Ambulance,Fire truck,Other"
```

You can also use the provided example script:

```bash
python example_two_step_pipeline.py \
    --data_dir /home/lkk/Developer/VisionLangAnnotate/output/dashcam_videos/onevideo_yolo11l_v2/20250613_061117_NF_20250710_135215 \
    --model_name Salesforce/blip2-opt-2.7b \
    --output_file example_results.json
```

## Output Format

The pipeline generates a JSON file with the following structure for each detected object:

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "bbox": [x1, y1, x2, y2],
    "step1_label": "car",
    "step1_score": 0.95,
    "vlm_description": "A blue sedan parked on the side of the road.",
    "ollama_class": "Car",
    "ollama_confidence": 0.92,
    "ollama_reasoning": "The image shows a four-wheeled passenger vehicle with a sedan body style, which matches the 'Car' class."
  },
  ...
]
```

Note: The `ollama_class`, `ollama_confidence`, and `ollama_reasoning` fields are only present when using the three-step pipeline with an Ollama model.

## How It Works

1. **Loading Annotations**: The pipeline loads Label Studio annotations from a JSON file.
2. **Extracting Bounding Boxes**: It extracts bounding box coordinates, converting from percentage-based to pixel-based coordinates.
3. **Grouping by Image**: Objects are grouped by image to avoid loading the same image multiple times.
4. **Generating Prompts**: Custom prompts are generated based on the object label (e.g., different prompts for cars, people, traffic signs).
5. **VLM Processing**: The HuggingFaceVLM model processes each cropped object with its corresponding prompt.
6. **Ollama Processing (Optional)**: If an Ollama model is specified, the VLM descriptions are processed to standardize outputs into predefined classes.
7. **Result Compilation**: Results are compiled into a structured format and saved to a JSON file.

## Customizing Prompts

You can customize the prompts used for different object types by modifying the `generate_prompt` function in the `two_step_pipeline.py` file. The current implementation includes specialized prompts for:

- People/pedestrians
- Vehicles (cars, trucks, buses)
- Bicycles and motorcycles
- Traffic lights and signs
- Other objects (default prompt)