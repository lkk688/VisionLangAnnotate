# VisionLangAnnotate

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</p>

## 🔍 Overview

VisionLangAnnotate is an advanced vision-language annotation framework that enables dynamic object detection and annotation based on natural language prompts. By combining traditional computer vision models with state-of-the-art Vision-Language Models (VLMs), it offers a flexible and powerful solution for object detection, segmentation, and annotation tasks.

## ✨ Key Features

- **Prompt-Driven Object Detection**: Detect objects based on natural language descriptions
- **Multi-Model Integration**: Combines traditional object detectors with Vision-Language Models
- **Dynamic Annotation**: Generate annotations on-the-fly based on user requests
- **Video Processing**: Support for both image and video inputs
- **Zero-Shot Capabilities**: Detect novel objects without prior training
- **Unified API**: Simple interface for various detection and annotation tasks
- **Label Studio Integration**: Export detection results to Label Studio compatible JSON format for human validation and re-annotation
- **Complete Annotation Loop**: End-to-end pipeline from user requests to data pre-processing, object detection, vision annotation, and human validation

## 🏗️ Architecture

VisionLangAnnotate consists of two main components:

### 1. Traditional Object Detection Pipeline

Located in `VisionLangAnnotateModels/detectors/`, this component includes:

- Multiple object detection models (DETR, YOLO, RT-DETR)
- Video processing pipeline (`videopipeline.py`)
- Ensemble detection capabilities
- Evaluation tools and metrics

### 2. Vision-Language Models (VLM) Pipeline

Located in `VisionLangAnnotateModels/VLM/`, this component includes:

- Zero-shot object detection with Grounding DINO and SAM (`groundingdinosam.py`)
- Multi-model VLM pipeline (`vlmpipeline.py`)
- Region captioning and classification
- Support for various VLM providers (OpenAI, Ollama, vLLM, LiteLLM)

### 3. Annotation and Validation Pipeline

The project includes tools for exporting detection results to Label Studio:

- Export detection results to Label Studio compatible JSON format (`export_to_label_studio.py`)
- Integration with Google Cloud Storage for data management (`utils/labelstudiogcp.py`)
- Complete workflow from automatic detection to human validation and re-annotation
- Feedback loop for continuous model improvement based on validated annotations

## 🚀 Usage Examples

### Object Detection with Natural Language

```python
from VisionLangAnnotateModels.VLM.vlmpipeline import VLMPipeline

pipeline = VLMPipeline()
results = pipeline.detect_with_prompt(
    image_path="sampledata/bus.jpg",
    prompt="Find all vehicles and count how many people are visible"
)

# Visualize results
pipeline.visualize_results(results, output_path="output.jpg")
```

### Zero-Shot Object Detection

```python
from VisionLangAnnotateModels.VLM.groundingdinosam import GroundingDinoSamDetector

# Initialize the zero-shot detector
detector = GroundingDinoSamDetector()

# Detect objects based on text prompt
results = detector.detect(
    image_path="path/to/image.jpg",
    text_prompt="Find a red car and a person wearing a hat"
)
```

### Video Processing

```python
from VisionLangAnnotateModels.detectors.videopipeline import VideoPipeline

# Initialize the video pipeline
video_pipeline = VideoPipeline(detector_name="yolov8x")

# Process a video file
detections = video_pipeline.process_video(
    video_path="path/to/video.mp4",
    output_path="path/to/output.mp4"
)
```

### Label Studio Export

```python
from VisionLangAnnotateModels.export_to_label_studio import export_detections_to_label_studio

# Export detection results to Label Studio format
export_detections_to_label_studio(
    detections=results,
    image_path="path/to/image.jpg",
    output_path="label_studio_annotations.json"
)

# For GCP integration
from VisionLangAnnotateModels.utils.labelstudiogcp import upload_to_gcs

# Upload annotations to Google Cloud Storage
upload_to_gcs(
    local_file_path="label_studio_annotations.json",
    bucket_name="your-bucket-name",
    destination_blob_name="annotations/label_studio_annotations.json"
)
```

## 🌆 Application: AI City Issue Detection System

The **AI City Issue Detection System** is a real-time monitoring solution that leverages the **VisionLangAnnotate** framework to automatically detect and annotate urban issues from city camera streams. By combining traditional deep learning-based object detection with **Vision-Language Models (VLMs)**, the system can identify a wide range of urban problems through natural language prompts and generate detailed annotations to support rapid response and resolution.

### 1. Data Acquisition Layer
- **Camera Stream Integration**: Connects to existing city surveillance cameras and traffic monitoring systems.
- **Video Processing Pipeline**: Extracts frames via scene change detection at configurable intervals for analysis and processes data in Google Cloud Storage buckets.
- **Edge Computing Support**: Deploys lightweight models on edge devices for initial filtering.
- **Privacy Protection**: Leverages deep learning models to automatically detect and blur human faces and license plate numbers in video streams.

### 2. Detection and Analysis Layer
- **Deep Learning-based Object Detection**: Identifies common urban objects such as vehicles, pedestrians, and infrastructure.
  - YOLO-based detectors for real-time performance.
  - DETR models for complex scene understanding.
  - Ensemble approaches to improve accuracy.
- **Vision-Language Model Processing**:
  - Zero-shot detection using Grounding DINO + SAM for novel issue types, with SAM providing precise object segmentation.
  - Natural language prompt processing (e.g., "Find potholes", "Detect broken streetlights") with VLM validation of detected object types, generation of detailed issue descriptions, and output in structured format for followup processing.
  - Region captioning to describe detected issues in detail.

### 3. Issue Classification and Prioritization
- **Issue Type Categorization**:
  - Infrastructure damage (e.g., potholes, cracks, broken facilities).
  - Traffic violations (e.g., illegal parking, wrong-way driving).
  - Safety hazards (e.g., fallen trees, flooding, debris).
  - Public space misuse (e.g., illegal dumping, graffiti).
- **Severity Assessment**:
  - Automatically prioritizes issues based on safety impact.
  - Tracks conditions over time to detect deterioration.

### 4. Annotation and Validation Loop
- **Automatic Annotation Generation**:
  - Generates bounding boxes and segmentation masks for detected issues.
  - Produces descriptive captions using VLMs.
  - Attaches metadata including location, timestamp, and severity.
- **Label Studio Integration**:
  - Exports annotations in Label Studio-compatible JSON format.
  - Supports human expert validation and correction workflow.
  - Incorporates a feedback loop for continuous model improvement.

### 5. Response and Monitoring System
- **Real-Time Alerts**:
  - Sends notifications for high-priority issues.
  - Integrates with city maintenance and management systems.
- **Issue Tracking Dashboard**:
  - Visualizes detected issues on a city map.
  - Supports historical data analysis and trend identification.
  - Tracks performance metrics and response times.

## 🛠️ Repository Setup
```bash
git clone https://github.com/lkk688/VisionLangAnnotate.git
cd VisionLangAnnotate
% conda env list
% conda activate mypy311
pip freeze > requirements.txt
pip install -r requirements.txt
#Install in Development Mode
#pip install -e .
pip install flit
flit install --symlink
#test import models: >>> import VisionLangAnnotateModels
```

Create Conda virtual environment and install cuda
```bash
conda create --name py312 python=3.12
conda activate py312
conda info --envs #check existing conda environment
% conda env list
$ conda install cuda -c nvidia/label/cuda-12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install mkdocs mkdocs-material
```

```bash
#test backend
uvicorn src.main:app --reload
#Verify at: http://localhost:8000
```

```bash
#Set Up React Frontend
cd ../frontend
brew install node
npm create vite@latest . --template react 
#choose react, JavaScript+SWC(Speedy Web Compiler) a Rust-based alternative to Babel.
npm install
#run the frontend
npm run dev
```

```bash
npm install @vitejs/plugin-react --save-dev
npm install react-router-dom
```

Documents
```bash
pip install mkdocs mkdocs-material
docs % mkdocs new .
docs % ls
docs                    getting-started.md      mkdocs.yml
#Run locally:
mkdocs serve --dev-addr localhost:8001  #Docs will be at: http://localhost:8001, default port is 8000
#find the process using port 8000
lsof -i :8000
#kill -9 <PID>
```

Git setup
```bash
git add .
git commit -m "Initial setup: FastAPI + React + Docs"
git push origin main
```

Dockerize: backend/Dockerfile
Backend: Deploy FastAPI (Render, Railway).
Frontend: Deploy React (Vercel, Netlify).

## GCP Setup
Install gcloud cli from https://cloud.google.com/sdk/docs/install-sdk#deb
```bash
gcloud init
gcloud auth login
gcloud config set project <project_id>
gcloud config set compute/zone <zone>
$ gcloud projects list
gsutil ls gs://roadsafetytarget/
gsutil ls "gs://roadsafetysource/Sweeper 19303/"
gcloud auth application-default login
```

## VLM Backends
Ollama installation: [ollama linux](https://ollama.com/download/linux)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gpt-oss:20b
ollama run gpt-oss:20b
```
Ollama exposes a Chat Completions-compatible API, so you can use the OpenAI SDK

## 🌐 Web Interface

VisionLangAnnotate provides a modern web interface built with React and FastAPI:

- **Backend**: FastAPI server with WebSocket support for real-time processing
- **Frontend**: React-based UI with intuitive controls for uploading media and entering prompts
- **API**: RESTful endpoints for programmatic access to all features

### Starting the Web Interface

```bash
# Start the backend server
uvicorn src.main:app --reload

# In a separate terminal, start the frontend
cd frontend
npm run dev
```

Visit `http://localhost:5173` to access the web interface.

## 📊 Evaluation and Metrics

VisionLangAnnotate includes tools for evaluating detection performance:

- Precision, Recall, and F1-score metrics
- Visualization of detection results
- Comparison between different models and approaches
- Benchmark datasets and evaluation scripts

## 🔄 Complete Annotation Loop

VisionLangAnnotate creates a complete annotation workflow:

1. **User Request**: Start with natural language prompts describing objects to detect
2. **Data Pre-processing**: Prepare images or videos for detection
3. **Automatic Detection**: Apply traditional detectors or VLMs based on the prompt
4. **Result Export**: Generate Label Studio compatible JSON format
5. **Human Validation**: Review and correct annotations in Label Studio
6. **Feedback Integration**: Use validated annotations to improve models
7. **Continuous Improvement**: Retrain or fine-tune models with validated data

This closed-loop system combines the efficiency of automatic detection with the accuracy of human validation, creating a powerful tool for building high-quality annotated datasets.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers directly.

---

<p align="center">Built with ❤️ for the computer vision and AI community</p>