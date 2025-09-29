# Enhanced VLM Utility Tutorial

A comprehensive guide to setting up, using, and troubleshooting vision-language models with the enhanced VLM_utils class, supporting multiple backends including vLLM and Ollama.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Backend Configuration](#backend-configuration)
4. [VLM_utils Architecture](#vlm_utils-architecture)
5. [Backend Comparison](#backend-comparison)
6. [Usage Examples](#usage-examples)
7. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Best Practices](#best-practices)

## Introduction

This tutorial covers the complete setup and usage of the enhanced `VLM_utils` class, which provides a unified interface for vision-language model operations across multiple backends. The class automatically handles backend selection, fallback mechanisms, and provides consistent functionality regardless of the underlying implementation.

### Supported Backends

The `VLM_utils` class supports three different backends:

- **vLLM API Backend**: Uses an existing vLLM server via HTTP API
- **vLLM Package Backend**: Uses the vLLM Python package directly
- **Ollama Backend**: Uses Ollama API for local model inference

### Key Features

- **Unified Interface**: Consistent API across all backends
- **Automatic Fallback**: Tries multiple backends until one succeeds
- **Backend Detection**: Automatically detects available backends
- **Enhanced Error Handling**: Comprehensive error reporting and recovery
- **Performance Metrics**: Detailed performance tracking for all backends

## Installation and Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with sufficient VRAM (for vLLM backends)
- Required Python dependencies

### Core Dependencies

```bash
# Essential packages for all backends
pip install requests pillow torch torchvision

# For logging and utilities
pip install typing-extensions
```

### Backend-Specific Installation

#### vLLM Installation

```bash
# Install vLLM with CUDA support
pip install vllm

# For development/testing
pip install vllm[dev]
```

#### Ollama Installation

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download

# Start Ollama service
ollama serve

# Pull a vision-language model
ollama pull llava
ollama pull qwen2-vl
```

### Verification

```bash
# Test vLLM installation
python -c "import vllm; print('vLLM installed successfully')"

# Test Ollama connection
curl http://localhost:11434/api/tags

# Test the VLM_utils class
python -c "from vlm_utils import VLMUtils; print('VLM_utils imported successfully')"
```

## Backend Configuration

### vLLM Server Configuration

#### Supported Model Variants

1. **Standard Model**: `Qwen/Qwen2.5-VL-7B-Instruct`
2. **AWQ Quantized**: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
3. **GPTQ Quantized**: `Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int4`
4. **FP8 Quantized**: `Qwen/Qwen2.5-VL-7B-Instruct-FP8`

#### Server Startup Commands

##### AWQ Quantized Model (Recommended for Memory Efficiency)
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --limit-mm-per-prompt image=1 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80
```

##### Standard Model
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --limit-mm-per-prompt image=1 \
    --dtype float16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85
```

### Ollama Configuration

#### Available Vision Models

```bash
# List available models
ollama list

# Pull specific vision models
ollama pull llava:latest
ollama pull llava:13b
ollama pull qwen2-vl:latest
ollama pull qwen2-vl:7b
```

#### Model Configuration

```bash
# Create custom model configuration (optional)
cat > Modelfile << EOF
FROM llava:latest
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
EOF

# Create custom model
ollama create my-vision-model -f Modelfile
```

### Memory Optimization Parameters

| Parameter | Description | vLLM Recommended | Ollama Equivalent |
|-----------|-------------|------------------|-------------------|
| Max sequence length | Maximum context window | 8192-16384 | Set via API |
| GPU memory usage | Memory utilization ratio | 0.80-0.85 | Automatic |
| Multimodal inputs | Images per prompt | `image=1` | Automatic |
| Precision | Model precision | `float16` | Automatic |

## VLM_utils Architecture

### Class Overview

The `VLM_utils` class is located at `/Developer/VisionLangAnnotate/tools/vlm_utils.py` and provides a unified interface for vision-language model operations across multiple backends. It implements a modular architecture with automatic backend detection and fallback mechanisms.

### Core Components

#### 1. Backend Enumeration
```python
class VLMBackend(Enum):
    """Enumeration for VLM backend types."""
    VLLM_API = "vllm_api"
    VLLM_PACKAGE = "vllm_package"
    OLLAMA = "ollama"
```

#### 2. Abstract Backend Interface
```python
class VLMBackendInterface(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the backend."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and ready."""
        pass
    
    @abstractmethod
    def process_image(self, image_input: Union[str, Image.Image], prompt: str, **kwargs) -> Dict[str, Any]:
        """Process a single image with the backend."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass
```

#### 3. Main VLMUtils Class
```python
class VLMUtils:
    """
    Enhanced VLM utility class with multi-backend support.
    
    Supports automatic backend detection, fallback mechanisms, and provides
    a unified interface for vision-language model operations.
    """
```

### Backend Implementations

#### 1. VLLMAPIBackend
- **Purpose**: Communicates with a running vLLM server via HTTP API
- **Initialization**: Checks server health and fetches model information
- **Features**: 
  - Automatic model name detection from server
  - Comprehensive error handling
  - Performance metrics tracking
  - Base64 image encoding with resizing

#### 2. VLLMPackageBackend
- **Purpose**: Uses the vLLM Python package directly
- **Initialization**: Creates LLM instance with optimized parameters
- **Features**:
  - Direct model control
  - Memory optimization settings
  - Multi-modal data handling
  - Token-level performance metrics

#### 3. OllamaBackend
- **Purpose**: Uses Ollama API for local model inference
- **Initialization**: Checks Ollama service and available models
- **Features**:
  - Automatic model selection from available vision models
  - Ollama-specific performance metrics
  - Base64 image encoding
  - Model configuration support

### Key Methods and APIs

#### Initialization
```python
def __init__(self, 
             backends: Optional[List[VLMBackend]] = None,
             vllm_api_url: str = "http://localhost:8000",
             vllm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
             ollama_api_url: str = "http://localhost:11434",
             ollama_model_name: str = "llava",
             **kwargs):
```

**Parameters:**
- `backends`: List of backends to try (auto-detects if None)
- `vllm_api_url`: vLLM server URL
- `vllm_model_name`: Model name for vLLM backends
- `ollama_api_url`: Ollama server URL
- `ollama_model_name`: Model name for Ollama backend
- `**kwargs`: Backend-specific configuration

#### Core Processing Methods

##### 1. Single Image Processing
```python
def process_image(self, 
                 image_input: Union[str, Image.Image], 
                 prompt: str = "Describe this image in detail.",
                 max_tokens: int = 1024, 
                 resize_for_context: bool = True,
                 **kwargs) -> Dict[str, Any]:
    """Process a single image using the first available backend."""
```

##### 2. Multiple Image Processing
```python
def process_multiple_images(self, 
                           image_inputs: List[Union[str, Image.Image]], 
                           prompts: Union[str, List[str]] = "Describe this image in detail.",
                           max_tokens: int = 1024, 
                           resize_for_context: bool = True,
                           **kwargs) -> List[Dict[str, Any]]:
    """Process multiple images using available backends."""
```

##### 3. Backend Management
```python
def check_server_status(self) -> Dict[str, Any]:
    """Check the status of all configured backends."""

def get_available_backends(self) -> List[VLMBackend]:
    """Get list of currently available backends."""

def switch_backend(self, backend: VLMBackend) -> bool:
    """Switch to a specific backend."""
```

### Image Processing Architecture

#### Universal Image Handling
The class provides consistent image processing across all backends:

- **Input Types**: File paths, URLs, PIL Image objects
- **Format Support**: JPEG, PNG, WebP, and other PIL-supported formats
- **Automatic Conversion**: Converts to RGB format as needed
- **Resizing**: Optional context-aware resizing for optimal performance

#### Token Constraints for Vision Models
- **Qwen2.5-VL**: Each 28x28 pixel patch = 1 visual token
- **Default range**: 256-1280 tokens (optimized for context window)
- **Automatic scaling**: Maintains aspect ratio while fitting constraints

#### Base64 Encoding
```python
def _encode_image_to_base64(self, image_input: Union[str, Image.Image], resize_for_context: bool = True) -> str:
    """Universal base64 encoding with optional resizing."""
```

### Response Format

All processing methods return a standardized dictionary:
```python
{
    "success": bool,             # Operation success status
    "response": str,             # Model response text
    "backend": str,              # Backend used for processing
    "performance": {             # Performance metrics (backend-specific)
        "duration_seconds": float,
        "tokens_per_second": float,
        # Additional backend-specific metrics
    },
    "error": str                 # Error message if any
}
```

### Fallback Mechanism

The class implements intelligent fallback:

1. **Backend Priority**: Tries backends in specified order
2. **Automatic Retry**: Switches to next backend on failure
3. **Error Aggregation**: Collects errors from all attempted backends
4. **Status Tracking**: Maintains backend availability status

## Usage Modes

The `VLM_utils` class supports three different backends, each with its own advantages:

### 1. Automatic Backend Detection (Recommended)

The simplest approach is to let the class automatically detect and use available backends:

```python
from tools.vlm_utils import VLMUtils

# Initialize with automatic backend detection
vlm = VLMUtils()

# Process a single image
result = vlm.process_image(
    image_input="path/to/image.jpg",
    prompt="Describe this image in detail.",
    max_tokens=1024
)

print(f"Backend used: {result['backend']}")
print(f"Response: {result['response']}")
```

### 2. vLLM API Mode (Server-based)

Use this mode when you have a vLLM server running:

```python
from tools.vlm_utils import VLMUtils, VLMBackend

# Initialize with vLLM API backend only
vlm = VLMUtils(
    backends=[VLMBackend.VLLM_API],
    vllm_api_url="http://localhost:8000",
    vllm_model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
)

# Check server status
status = vlm.check_server_status()
print(f"Server status: {status}")

# Process image
result = vlm.process_image(
    image_input="https://example.com/image.jpg",
    prompt="What objects can you see in this image?",
    max_tokens=512
)
```

### 3. vLLM Package Mode (Direct)

Use this mode for direct model control with the vLLM Python package:

```python
from tools.vlm_utils import VLMUtils, VLMBackend

# Initialize with vLLM package backend
vlm = VLMUtils(
    backends=[VLMBackend.VLLM_PACKAGE],
    vllm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    gpu_memory_utilization=0.8,
    max_model_len=8192
)

# Process image with custom parameters
result = vlm.process_image(
    image_input="path/to/image.png",
    prompt="Analyze the technical aspects of this image.",
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9
)
```

### 4. Ollama Mode

Use this mode for local inference with Ollama:

```python
from tools.vlm_utils import VLMUtils, VLMBackend

# Initialize with Ollama backend
vlm = VLMUtils(
    backends=[VLMBackend.OLLAMA],
    ollama_api_url="http://localhost:11434",
    ollama_model_name="qwen2-vl:7b"
)

# Process image
result = vlm.process_image(
    image_input="path/to/image.jpg",
    prompt="Describe this image in detail.",
    max_tokens=1024
)
```

### 5. Multi-Backend with Fallback

Configure multiple backends with automatic fallback:

```python
from tools.vlm_utils import VLMUtils, VLMBackend

# Initialize with preferred backend order
vlm = VLMUtils(
    backends=[VLMBackend.VLLM_API, VLMBackend.OLLAMA, VLMBackend.VLLM_PACKAGE],
    vllm_api_url="http://localhost:8000",
    ollama_api_url="http://localhost:11434",
    vllm_model_name="Qwen/Qwen2.5-VL-7B-Instruct"
)

# The class will try vLLM API first, then Ollama, then vLLM package
result = vlm.process_image("path/to/image.jpg", "Describe this image.")
print(f"Successfully used backend: {result['backend']}")
```

## Batch Processing

### Processing Multiple Images

```python
from tools.vlm_utils import VLMUtils

vlm = VLMUtils()

# Process multiple images with the same prompt
image_paths = ["image1.jpg", "image2.png", "image3.webp"]
results = vlm.process_multiple_images(
    image_inputs=image_paths,
    prompts="What is the main subject of this image?",
    max_tokens=512
)

# Process multiple images with different prompts
prompts = [
    "Describe the colors in this image.",
    "What emotions does this image convey?",
    "List all objects visible in this image."
]
results = vlm.process_multiple_images(
    image_inputs=image_paths,
    prompts=prompts,
    max_tokens=1024
)

# Print results
for i, result in enumerate(results):
    print(f"Image {i+1} ({result['backend']}): {result['response']}")
```

### Performance Monitoring

```python
from tools.vlm_utils import VLMUtils

vlm = VLMUtils()

# Process image and get performance metrics
result = vlm.process_image("path/to/image.jpg", "Describe this image.")

print(f"Processing time: {result['performance']['duration_seconds']:.2f}s")
print(f"Tokens per second: {result['performance']['tokens_per_second']:.2f}")

# Get performance summary across multiple images
results = vlm.process_multiple_images(
    image_inputs=["img1.jpg", "img2.jpg", "img3.jpg"],
    prompts="Describe this image."
)

# Calculate average performance
total_time = sum(r['performance']['duration_seconds'] for r in results)
avg_tps = sum(r['performance']['tokens_per_second'] for r in results) / len(results)
print(f"Total processing time: {total_time:.2f}s")
print(f"Average tokens per second: {avg_tps:.2f}")
```

## Backend Management

### Checking Backend Status

```python
from tools.vlm_utils import VLMUtils

vlm = VLMUtils()

# Check status of all configured backends
status = vlm.check_server_status()
print("Backend Status:")
for backend, info in status.items():
    print(f"  {backend}: {'✓' if info['available'] else '✗'}")
    if info.get('error'):
        print(f"    Error: {info['error']}")

# Get list of available backends
available = vlm.get_available_backends()
print(f"Available backends: {[b.value for b in available]}")
```

### Switching Backends

```python
from tools.vlm_utils import VLMUtils, VLMBackend

vlm = VLMUtils()

# Try to switch to a specific backend
if vlm.switch_backend(VLMBackend.OLLAMA):
    print("Successfully switched to Ollama backend")
    result = vlm.process_image("image.jpg", "Describe this image.")
else:
    print("Failed to switch to Ollama backend")
```

## Advanced Configuration

### Custom Image Processing

```python
from tools.vlm_utils import VLMUtils
from PIL import Image

vlm = VLMUtils()

# Load and preprocess image
image = Image.open("path/to/image.jpg")
image = image.convert("RGB")
image = image.resize((800, 600))

# Process PIL Image object directly
result = vlm.process_image(
    image_input=image,
    prompt="Analyze this preprocessed image.",
    resize_for_context=False  # Skip automatic resizing
)
```

### Backend-Specific Parameters

```python
from tools.vlm_utils import VLMUtils, VLMBackend

# vLLM API with custom parameters
vlm_api = VLMUtils(
    backends=[VLMBackend.VLLM_API],
    vllm_api_url="http://localhost:8000"
)

result = vlm_api.process_image(
    image_input="image.jpg",
    prompt="Describe this image.",
    temperature=0.8,
    top_p=0.95,
    frequency_penalty=0.1
)

# Ollama with custom parameters
vlm_ollama = VLMUtils(
    backends=[VLMBackend.OLLAMA],
    ollama_model_name="qwen2-vl:7b"
)

result = vlm_ollama.process_image(
    image_input="image.jpg",
    prompt="Describe this image.",
    temperature=0.7,
    num_predict=1024
)
```

## Backend Comparison

The `VLM_utils` class supports three different backends, each optimized for different use cases:

### Feature Comparison

| Feature | vLLM API | vLLM Package | Ollama |
|---------|----------|--------------|--------|
| **Setup Complexity** | Medium | High | Low |
| **Resource Usage** | Low (client) | High | Medium |
| **Performance** | High | Highest | Medium |
| **Scalability** | Excellent | Limited | Good |
| **Model Support** | vLLM models | vLLM models | Ollama models |
| **Customization** | Limited | Full | Medium |
| **Memory Efficiency** | Excellent | Variable | Good |

### Use Case Recommendations

#### 1. vLLM API Backend
**Best for**: Production environments, multi-user applications, resource-constrained clients

**Advantages**:
- Shared server resources across multiple clients
- Low client-side memory usage
- Easy horizontal scaling
- Automatic model management
- Built-in load balancing capabilities

**Disadvantages**:
- Requires separate server setup
- Network latency overhead
- Limited parameter customization per request

**Ideal scenarios**:
- Web applications serving multiple users
- Microservices architecture
- Cloud deployments
- When you need to share expensive GPU resources

#### 2. vLLM Package Backend
**Best for**: Research, development, maximum performance requirements

**Advantages**:
- Direct model control and customization
- No network overhead
- Full access to vLLM parameters
- Highest possible performance
- Custom sampling strategies

**Disadvantages**:
- High memory requirements
- Complex setup and configuration
- Not suitable for concurrent users
- Requires significant GPU resources

**Ideal scenarios**:
- Research experiments
- Batch processing large datasets
- Development and testing
- When you need maximum control over inference

#### 3. Ollama Backend
**Best for**: Local development, quick prototyping, ease of use

**Advantages**:
- Extremely simple setup
- Good model ecosystem
- Reasonable performance
- Built-in model management
- Cross-platform compatibility

**Disadvantages**:
- Limited to Ollama-supported models
- Less customization than vLLM
- Moderate performance compared to vLLM
- Fewer advanced features

**Ideal scenarios**:
- Local development and testing
- Proof of concepts
- Educational purposes
- When simplicity is prioritized over performance

### Performance Characteristics

#### Throughput Comparison
```
vLLM Package > vLLM API > Ollama
```

#### Memory Usage
```
vLLM Package (High) > Ollama (Medium) > vLLM API (Low - client side)
```

#### Setup Time
```
Ollama (Minutes) < vLLM API (Hours) < vLLM Package (Hours)
```

### Model Support Matrix

| Backend | Qwen2.5-VL | LLaVA | Other Vision Models |
|---------|------------|-------|-------------------|
| **vLLM API** | ✅ Full | ✅ Full | ✅ vLLM supported |
| **vLLM Package** | ✅ Full | ✅ Full | ✅ vLLM supported |
| **Ollama** | ✅ Limited | ✅ Full | ✅ Ollama supported |

### Fallback Strategy

The `VLM_utils` class implements intelligent fallback:

1. **Primary**: Try the first specified backend
2. **Secondary**: Fall back to next available backend
3. **Tertiary**: Continue until all backends are exhausted
4. **Error Handling**: Aggregate errors from all attempts

**Recommended fallback order**:
```python
# For production
backends = [VLMBackend.VLLM_API, VLMBackend.OLLAMA]

# For development
backends = [VLMBackend.OLLAMA, VLMBackend.VLLM_PACKAGE]

# For maximum reliability
backends = [VLMBackend.VLLM_API, VLMBackend.OLLAMA, VLMBackend.VLLM_PACKAGE]
```

## Common Issues and Troubleshooting

The `VLM_utils` class provides comprehensive error handling and fallback mechanisms. Here are common issues and their solutions:

### 1. Backend Initialization Issues

#### vLLM API Backend Issues

**Problem**: `ConnectionError: Cannot connect to vLLM server`

**Diagnosis**:
```python
from tools.vlm_utils import VLMUtils, VLMBackend

vlm = VLMUtils(backends=[VLMBackend.VLLM_API])
status = vlm.check_server_status()
print(status)
```

**Solutions**:
- Verify server is running: `ps aux | grep vllm`
- Check server health: `curl http://localhost:8000/health`
- Verify correct host/port configuration
- Check server logs for errors

#### vLLM Package Backend Issues

**Problem**: `ValueError: No available memory for the cache blocks`

**Cause**: Insufficient GPU memory for KV cache allocation.

**Solutions**:
```python
# Reduce memory usage
vlm = VLMUtils(
    backends=[VLMBackend.VLLM_PACKAGE],
    gpu_memory_utilization=0.70,  # Lower utilization
    max_model_len=4096,           # Reduce context length
    enforce_eager=True            # Disable CUDA graphs
)
```

**Alternative approach**:
```bash
# Use quantized models
export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
```

#### Ollama Backend Issues

**Problem**: `ConnectionError: Ollama service not available`

**Solutions**:
```bash
# Start Ollama service
ollama serve

# Check available models
ollama list

# Pull required model
ollama pull qwen2-vl:7b
```

**Problem**: `Model not found: llava`

**Solutions**:
```python
# Specify available model
vlm = VLMUtils(
    backends=[VLMBackend.OLLAMA],
    ollama_model_name="qwen2-vl:7b"  # Use available model
)

# Or let the class auto-detect
vlm = VLMUtils(backends=[VLMBackend.OLLAMA])
# The class will automatically find available vision models
```

### 2. Multi-Backend Fallback Issues

**Problem**: All backends fail to initialize

**Diagnosis**:
```python
vlm = VLMUtils()
status = vlm.check_server_status()
print("Backend Status:")
for backend, info in status.items():
    print(f"  {backend}: {'✓' if info['available'] else '✗'}")
    if info.get('error'):
        print(f"    Error: {info['error']}")
```

**Solutions**:
- Ensure at least one backend service is running
- Check individual backend configurations
- Use specific backend initialization for debugging

### 3. Image Processing Errors

**Problem**: Image loading or processing failures

**Common Causes and Solutions**:

#### File Access Issues:
```python
# Check file existence
import os
if not os.path.exists("path/to/image.jpg"):
    print("Image file not found")

# Use absolute paths
result = vlm.process_image(
    image_input=os.path.abspath("image.jpg"),
    prompt="Describe this image."
)
```

#### Format Issues:
```python
# Convert unsupported formats
from PIL import Image

image = Image.open("image.webp")
image = image.convert("RGB")
result = vlm.process_image(image_input=image, prompt="Describe this image.")
```

#### Large Image Issues:
```python
# Enable automatic resizing
result = vlm.process_image(
    image_input="large_image.jpg",
    prompt="Describe this image.",
    resize_for_context=True  # Automatically resize for optimal processing
)
```

### 4. Performance Issues

**Problem**: Slow inference or high memory usage

**Backend-Specific Optimizations**:

#### vLLM API:
```python
# Use streaming for long responses
result = vlm.process_image(
    image_input="image.jpg",
    prompt="Provide a detailed analysis.",
    max_tokens=512,  # Limit response length
    temperature=0.7
)
```

#### vLLM Package:
```python
# Optimize memory usage
vlm = VLMUtils(
    backends=[VLMBackend.VLLM_PACKAGE],
    gpu_memory_utilization=0.80,
    max_model_len=8192,
    enforce_eager=True,           # Disable CUDA graphs for memory
    disable_log_stats=True,       # Reduce logging overhead
    disable_custom_all_reduce=True
)
```

#### Ollama:
```python
# Optimize Ollama performance
result = vlm.process_image(
    image_input="image.jpg",
    prompt="Describe this image.",
    num_predict=512,    # Limit response length
    temperature=0.7,
    top_p=0.9
)
```

### 5. Model-Specific Issues

**Problem**: Model name mismatch or unsupported model

**vLLM Backends**:
```python
# Check available models
curl http://localhost:8000/v1/models

# The class automatically detects the correct model name
vlm = VLMUtils(backends=[VLMBackend.VLLM_API])
# Model name is automatically fetched from server
```

**Ollama Backend**:
```bash
# List available models
ollama list

# Pull specific vision model
ollama pull qwen2-vl:7b
ollama pull llava:latest
```

### 6. Debugging Multi-Backend Issues

**Enable detailed logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

vlm = VLMUtils()
result = vlm.process_image("image.jpg", "Describe this image.")
```

**Test individual backends**:
```python
# Test each backend separately
backends_to_test = [VLMBackend.VLLM_API, VLMBackend.OLLAMA, VLMBackend.VLLM_PACKAGE]

for backend in backends_to_test:
    try:
        vlm = VLMUtils(backends=[backend])
        result = vlm.process_image("test_image.jpg", "Test prompt")
        print(f"{backend.value}: ✓ Working")
    except Exception as e:
        print(f"{backend.value}: ✗ Error - {e}")
```

**Check backend availability**:
```python
vlm = VLMUtils()
available = vlm.get_available_backends()
print(f"Available backends: {[b.value for b in available]}")

if not available:
    print("No backends available. Check your setup:")
    print("1. vLLM server: curl http://localhost:8000/health")
    print("2. Ollama service: ollama list")
    print("3. GPU memory for vLLM package mode")
```

### 7. Common Error Messages and Solutions

| Error Message | Backend | Solution |
|---------------|---------|----------|
| `No available memory for the cache blocks` | vLLM Package | Reduce `gpu_memory_utilization` or `max_model_len` |
| `Connection refused` | vLLM API | Start vLLM server |
| `Model not found` | All | Check model name and availability |
| `Ollama service not available` | Ollama | Start Ollama service with `ollama serve` |
| `No backends available` | All | Ensure at least one backend service is running |
| `Image processing failed` | All | Check image format and file accessibility |

### 8. Performance Optimization Tips

#### Memory Optimization:
```python
# Conservative settings for limited GPU memory
vlm = VLMUtils(
    backends=[VLMBackend.VLLM_API, VLMBackend.OLLAMA],  # Avoid package mode
    vllm_api_url="http://localhost:8000",
    ollama_api_url="http://localhost:11434"
)
```

#### Batch Processing:
```python
# Process multiple images efficiently
results = vlm.process_multiple_images(
    image_inputs=["img1.jpg", "img2.jpg", "img3.jpg"],
    prompts="Describe this image.",
    max_tokens=512  # Limit response length for faster processing
)
```

#### Image Preprocessing:
```python
# Preprocess images for optimal performance
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    # Resize to reasonable dimensions
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return image

result = vlm.process_image(
    image_input=preprocess_image("large_image.jpg"),
    prompt="Describe this image.",
    resize_for_context=False  # Skip automatic resizing
)
    image_paths,
    prompts,
    max_tokens=512,
    resize_for_context=True
)
```

## Examples and Best Practices

### Example 1: Basic Image Description
```python
from vlm_vllm_utils import VLMVLLMUtility, VLMMode

# Initialize utility
vlm = VLMVLLMUtility(mode=VLMMode.URL)

# Process image
result = vlm.process_image(
    "path/to/image.jpg",
    "Describe what you see in this image in detail.",
    max_tokens=1024
)

if result["success"]:
    print(f"Response: {result['response']}")
    print(f"Processing time: {result['performance']['duration_seconds']:.2f}s")
    print(f"Tokens per second: {result['performance']['tokens_per_second']:.1f}")
else:
    print(f"Error: {result['error']}")
```

### Example 2: Batch Processing
```python
# Process multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
prompts = [
    "What objects are in this image?",
    "Describe the scene and setting.",
    "What activities are taking place?"
]

results = vlm.process_images_batch(
    image_paths,
    prompts,
    max_tokens=512,
    resize_for_context=True
)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['response']}")
```

### Example 3: Server Status Monitoring
```python
# Check server health and model information
status = vlm.check_server_status()

if status["server_running"]:
    print("✅ Server is running")
    print(f"Available models: {[model['id'] for model in status['models']]}")
else:
    print(f"❌ Server issue: {status['error']}")
```

### Example 4: Error Handling
```python
def safe_process_image(vlm, image_path, prompt):
    """Safely process an image with comprehensive error handling."""
    try:
        result = vlm.process_image(image_path, prompt)
        
        if result["success"]:
            return result["response"]
        else:
            print(f"Processing failed: {result['error']}")
            return None
            
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

# Usage
response = safe_process_image(vlm, "image.jpg", "What do you see?")
if response:
    print(response)
```

### Best Practices

1. **Always enable automatic resizing** for optimal performance:
   ```python
   resize_for_context=True
   ```

2. **Use appropriate model variants** based on your hardware:
   - AWQ for balanced performance/quality
   - GPTQ for maximum compression
   - Standard for best quality (if VRAM allows)

3. **Monitor performance metrics**:
   ```python
   perf = result["performance"]
   print(f"Speed: {perf['tokens_per_second']:.1f} tokens/sec")
   ```

4. **Implement proper error handling** for production use

5. **Use batch processing** for multiple images to improve efficiency

6. **Check server status** before processing in URL mode

7. **Configure memory parameters** based on your GPU capabilities

## Conclusion

This tutorial provides a comprehensive guide to using vLLM with Qwen2.5-VL models through the `VLMVLLMUtility` class. The utility provides a robust, flexible interface that handles common issues automatically while offering both server-based and direct package modes for different use cases.

Key takeaways:
- Use URL mode for production deployments
- Use Package mode for development and testing
- Enable automatic image resizing for optimal performance
- Monitor memory usage and adjust parameters accordingly
- Implement proper error handling for robust applications

For additional support and updates, refer to the vLLM documentation and the Qwen2.5-VL model documentation.