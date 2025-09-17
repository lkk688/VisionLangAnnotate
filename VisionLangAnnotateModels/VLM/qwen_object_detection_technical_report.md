# Technical Report: Qwen Object Detection Pipeline

## Executive Summary

The `QwenObjectDetectionPipeline` represents a significant advancement in object detection technology by implementing a novel hybrid approach that combines traditional computer vision object detectors with Vision-Language Models (VLMs). This innovative pipeline leverages the strengths of both paradigms to achieve superior detection performance, enhanced object description capabilities, and improved robustness across diverse visual scenarios.

## Key Innovations

### 1. Four Detection Comparison Modes

The core innovation of this pipeline is its **four distinct detection modes** that provide comprehensive comparison between traditional and VLM-based approaches:

#### Mode 1: VLM-Only Detection
Pure Vision-Language Model detection using Qwen2.5-VL without traditional detectors:
- **Natural Language Understanding**: Leverages Qwen2.5-VL's ability to understand and describe visual content
- **Zero-Shot Detection**: Identifies objects without prior training on specific categories
- **Rich Object Descriptions**: Provides detailed natural language descriptions of detected objects

#### Mode 2: Standard Hybrid Detection
Combines traditional detectors with VLM in parallel processing:
- **Multi-Model Support**: Incorporates YOLO, DETR, and RT-DETR models through a unified interface
- **Ensemble Detection**: Combines results from multiple traditional detectors using advanced fusion techniques
- **Class Mapping**: Intelligently maps COCO classes to domain-specific categories
- **Parallel Processing**: Both traditional and VLM detectors run simultaneously on the full image

#### Mode 3: Sequential Hybrid Detection
Sequential processing where traditional detectors run first, followed by VLM enhancement:
- **Two-Stage Pipeline**: Traditional detectors provide initial detections, VLM enhances with descriptions
- **Confidence Boosting**: VLM can validate and enhance traditional detector results
- **Complementary Detection**: VLM can identify objects missed by traditional detectors

#### Mode 4: Cropped Sequential Hybrid Detection
Advanced sequential mode with region-based VLM processing:
- **Region-Based Analysis**: VLM processes cropped regions from traditional detector bounding boxes
- **Enhanced Accuracy**: Focused analysis on specific regions improves detection precision
- **Detailed Descriptions**: VLM provides more accurate descriptions for cropped object regions
- **Hierarchical Processing**: Combines global and local analysis for comprehensive detection

### 2. Hybrid Fusion Strategies

The pipeline implements multiple fusion strategies for combining detections:

1. **Ensemble Mode**: Combines all detections using Non-Maximum Suppression (NMS) or Weighted Box Fusion (WBF)
2. **VLM-Enhanced Mode**: Uses traditional detectors for localization but enhances with VLM descriptions
3. **Complementary Mode**: Uses VLM to detect objects missed by traditional detectors
4. **Box Optimization Mode**: Merges and refines bounding boxes from all sources

## Comparative Analysis of Detection Modes

The pipeline's four detection modes provide comprehensive comparison capabilities for evaluating different approaches:

### Performance Comparison Framework

```python
# VLM-only detection
results_vlm = pipeline.detect_objects(image_path, use_sam_segmentation=True)
print(f"VLM-only detection: {len(results_vlm['objects'])} objects")

# Standard hybrid mode (parallel processing)
results_hybrid = pipeline.detect_objects_hybrid(
    image_path, 
    use_sam_segmentation=True, 
    sequential_mode=False, 
    cropped_sequential_mode=False, 
    save_results=True
)
print(f"Hybrid detection: {len(results_hybrid['objects'])} objects")
print(f"Traditional detectors used: {results_hybrid['traditional_detectors_used']}")

# Sequential hybrid mode
results_hybrid_seq = pipeline.detect_objects_hybrid(
    image_path, 
    use_sam_segmentation=True, 
    sequential_mode=True, 
    cropped_sequential_mode=False, 
    save_results=True
)
print(f"Hybrid detection Sequential mode: {len(results_hybrid_seq['objects'])} objects")

# Cropped Sequential hybrid mode
results_hybrid_cropped = pipeline.detect_objects_hybrid(
    image_path, 
    use_sam_segmentation=True, 
    sequential_mode=True, 
    cropped_sequential_mode=True, 
    save_results=True
)
print(f"Hybrid detection Cropped Sequential mode: {len(results_hybrid_cropped['objects'])} objects")
```

Use sample image "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/sj1.jpg" as one example. The complete detection results is shown in `results/qwen_detection_results`

The object detection results of `Qwen2.5-VL only` is shown in the image below:
![Detection Results](/results/qwen_detection_results/visualizations/sj1_20250917_102554_detection.png)
The segmentation results is shown in the image below:
![Segmentation Results](results/qwen_detection_results/segmentations/sj1_20250917_102554_segmentation.png)


The object detetion results of hybrid mode 1 `(sequential_mode=False, cropped_sequential_mode=False)` is shown in the image below:
![Detection Results](/results/qwen_detection_results/visualizations/sj1_20250917_102850_detection.png)
The segmentation results of hybrid mode 1 is shown in the image below:
![Segmentation Results](results/qwen_detection_results/segmentations/sj1_20250917_102850_hybrid_segmentation.png)

The object detetion results of hybrid mode 2 `(sequential_mode=True, cropped_sequential_mode=False)` is shown in the image below:
![Detection Results](results/qwen_detection_results/visualizations/sj1_20250917_103044_detection.png)
The segmentation results of hybrid mode 2 is shown in the image below:
![Segmentation Results](results/qwen_detection_results/segmentations/sj1_20250917_103043_hybrid_segmentation.png)

The object detection results of hybrid mode 3 `(cropped_sequential_mode=True)` is shown in the image below:
![Detection Results](results/qwen_detection_results/visualizations/sj1_20250917_103220_detection.png)
The segmentation results of hybrid mode 3 is shown in the image below:
![Segmentation Results](results/qwen_detection_results/segmentations/sj1_20250917_103220_hybrid_segmentation.png)

### Mode-Specific Advantages

| Mode | Strengths | Use Cases |
|------|-----------|-----------|
| **VLM-Only** | Zero-shot detection, rich descriptions, novel object recognition | Research, exploratory analysis, unknown domains |
| **Standard Hybrid** | Fast parallel processing, ensemble robustness | Real-time applications, balanced accuracy |
| **Sequential Hybrid** | Enhanced accuracy, validation pipeline | High-precision requirements, quality assurance |
| **Cropped Sequential** | Maximum detail, region-focused analysis | Fine-grained detection, detailed object analysis |

### 3. Natural Language Parsing Engine

The pipeline implements a sophisticated regex-based parsing engine that extracts structured object detection information from VLM's natural language responses, supporting multiple response formats for robust parsing across different VLM outputs.

### 4. Box Optimization Algorithms

The pipeline introduces novel algorithms for optimizing bounding boxes to improve detection quality:

- **IoU-based Merging**: Combines overlapping detections using configurable thresholds
- **Confidence Weighting**: Prioritizes high-confidence detections during optimization
- **Multi-source Integration**: Intelligently merges boxes from traditional and VLM detectors

### 5. Adaptive Prompting System

The pipeline implements an adaptive prompting system that generates context-aware prompts for the VLM:

```python
def _generate_detection_prompt(self, image_path, prompt_type='detailed'):
    """Generate an appropriate prompt for object detection based on prompt type."""
    if prompt_type == 'detailed':
        return f"""Analyze this image and detect all objects. 
        For each object, provide:
        1. Object name
        2. Bounding box coordinates [x1, y1, x2, y2]
        3. Brief description
        
        Format your response as a JSON with an 'objects' array containing each detection.
        Example: {{"objects": [{{"label": "car", "bbox": [100, 150, 300, 250], "description": "red sedan"}}]}}
        """
    elif prompt_type == 'simple':
        return "What objects do you see in this image? List each with its bounding box coordinates [x1, y1, x2, y2]."
    else:
        return "Detect and describe all objects in this image with their bounding box coordinates [x1, y1, x2, y2]."
```


## Technical Implementation Details

### Ensemble Methods

The pipeline implements two advanced ensemble methods for combining detections:

1. **Non-Maximum Suppression (NMS)** - See <mcsymbol name="_nms_ensemble_detections" filename="qwen_object_detection_pipeline2.py" path="VisionLangAnnotateModels/VLM/qwen_object_detection_pipeline2.py" startline="2845" type="function"></mcsymbol>

2. **Weighted Box Fusion (WBF)** - See <mcsymbol name="_wbf_ensemble_detections" filename="qwen_object_detection_pipeline2.py" path="VisionLangAnnotateModels/VLM/qwen_object_detection_pipeline2.py" startline="2885" type="function"></mcsymbol>

### Hybrid Object Creation

The pipeline implements a sophisticated hybrid object creation process that combines information from both traditional detectors and VLMs. See <mcsymbol name="ensemble_hybrid_vlm_detections" filename="qwen_object_detection_pipeline2.py" path="VisionLangAnnotateModels/VLM/qwen_object_detection_pipeline2.py" startline="2950" type="function"></mcsymbol> for the complete implementation.

Key features:
- **Step 1**: Ensemble all detections (traditional + VLM)
- **Step 2**: Optimize boxes (merge small/nearby boxes)  
- **Step 3**: Convert to final format with hybrid logic:
  - Prioritize VLM object names and descriptions
  - Use traditional detector bounding boxes when available
  - For objects not detected by VLM, use original COCO class with no description

### Box Optimization

The pipeline implements advanced box optimization techniques to improve detection quality. See <mcsymbol name="optimize_boxes_for_vlm" filename="qwen_object_detection_pipeline2.py" path="VisionLangAnnotateModels/VLM/qwen_object_detection_pipeline2.py" startline="3070" type="function"></mcsymbol> for the complete implementation.

Key optimization features:
- **Size Filtering**: Removes very small boxes based on minimum size requirements
- **Confidence Sorting**: Prioritizes high-confidence detections
- **Proximity Merging**: Merges nearby boxes using IoU and distance metrics
- **Box Limit Management**: Maintains optimal number of boxes for processing

## Performance Capabilities

The `QwenObjectDetectionPipeline` demonstrates several key performance capabilities:

1. **Robust Detection**: Successfully detects objects across diverse visual scenarios
2. **Enhanced Object Descriptions**: Provides rich natural language descriptions of detected objects
3. **Flexible Configuration**: Supports multiple detection modes and ensemble methods
4. **Fault Tolerance**: Gracefully handles missing dependencies and model failures
5. **Visualization**: Generates high-quality visualizations of detection results

## Current work

TBD