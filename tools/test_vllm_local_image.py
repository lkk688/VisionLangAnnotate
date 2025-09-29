#!/usr/bin/env python3
"""
Example of using vLLM API with local image files.
Requires vLLM server to be running:
GPTQ (General Post-Training Quantization)
AWQ (Activation-aware Weight Quantization)

INT4 version (GPTQ): Qwen/Qwen2.5-VL-7B-Instruct-AWQ
INT4 version (AWQ): Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int4
RedHatAI/Qwen2.5-VL-7B-Instruct-FP8-Dynamic

$ vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --limit-mm-per-prompt '{"image": 5}' \
    --max-model-len 32768

vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --limit-mm-per-prompt '{"image": 5}' \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80

vllm serve RedHatAI/Qwen2.5-VL-7B-Instruct-FP8-Dynamic \
    --host 0.0.0.0 \
    --port 8000 \
    --limit-mm-per-prompt '{"image": 5}' \
    --quantization fp8

vllm serve Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int4 \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization gptq \
    --limit-mm-per-prompt '{"image": 5}' \
    --max-model-len 32768

"""

#Qwen2.5-VL uses "dynamic resolution" processing, which means it can handle images of various sizes and automatically converts them into appropriate token sequences
# Theoretical examples:
# ==================================================
# Small image      400x 300 ( 120,000 px) -> RESIZE UP to 517x387 (255 tokens)
# Medium image     800x 600 ( 480,000 px) -> NO RESIZE (612 tokens)
# Large image     1920x1080 (2,073,600 px) -> RESIZE DOWN to 1335x751 (1,278 tokens)
# Very large image 4000x3000 (12,000,000 px) -> RESIZE DOWN to 1156x867 (1,278 tokens)
# Ultra high-res  8000x6000 (48,000,000 px) -> RESIZE DOWN to 1156x867 (1,278 tokens)

import requests
import json
import base64
import os
from PIL import Image
import io
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import statistics
import urllib.request
import urllib.parse

def detect_image_input_type(image_input: Union[str, Image.Image]) -> str:
    """
    Detect the type of image input: URL, file path, or PIL Image.
    
    Args:
        image_input: Can be a URL string, file path string, or PIL Image object
        
    Returns:
        String indicating the input type: 'url', 'file_path', or 'pil_image'
    """
    if isinstance(image_input, Image.Image):
        return 'pil_image'
    elif isinstance(image_input, str):
        # Check if it's a URL
        parsed = urllib.parse.urlparse(image_input)
        if parsed.scheme in ('http', 'https'):
            return 'url'
        else:
            return 'file_path'
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

def download_image_from_url(url: str) -> Image.Image:
    """
    Download an image from a URL and return as PIL Image.
    
    Args:
        url: HTTP/HTTPS URL to the image
        
    Returns:
        PIL Image object
    """
    try:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
            return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Failed to download image from URL {url}: {str(e)}")

def resize_image_for_context(image_input: Union[str, Image.Image], max_pixels=1280*28*28, min_pixels=256*28*28):
    """
    Resize image to fit within token constraints for Qwen2.5-VL-7B-Instruct.
    
    With max-model-len=32768:
    - Each 28x28 pixel patch = 1 visual token
    - Default range: 4-16384 visual tokens per image
    - Recommended range: 256-1280 tokens (256*28*28 to 1280*28*28 pixels)
    - This leaves room for text tokens in the 32K context window
    
    Args:
        image_input: Can be a file path (str), URL (str), or PIL Image object
        max_pixels: Maximum pixels (default: 1280*28*28 = 1,003,520 pixels)
        min_pixels: Minimum pixels (default: 256*28*28 = 200,704 pixels)
    
    Returns:
        PIL Image object resized to fit constraints
    """
    # Handle different input types
    input_type = detect_image_input_type(image_input)
    
    if input_type == 'pil_image':
        img = image_input.copy()
    elif input_type == 'url':
        img = download_image_from_url(image_input)
    else:  # file_path
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img = Image.open(image_input)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    current_pixels = width * height
    #image dimensions do NOT need to be exact multiples of 28 pixels
    #Automatic Rounding : The model automatically rounds dimensions to the nearest multiple of 28 during preprocessing

    # If image is within bounds, return as-is
    if min_pixels <= current_pixels <= max_pixels:
        return img.copy()
    
    # Calculate scaling factor
    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5
    else:  # current_pixels < min_pixels
        scale_factor = (min_pixels / current_pixels) ** 0.5
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

def encode_image_to_base64(image_input: Union[str, Image.Image], resize_for_context=True):
    """Convert image input (URL, file path, or PIL Image) to base64 data URL with optional resizing"""
    
    # Handle different input types
    input_type = detect_image_input_type(image_input)
    
    if resize_for_context:
        # Resize image to fit context constraints
        img = resize_image_for_context(image_input)
        
        # Convert PIL image to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = 'image/jpeg'
    else:
        # Handle different input types without resizing
        if input_type == 'pil_image':
            # Convert PIL image to base64
            buffer = io.BytesIO()
            image_input.save(buffer, format='JPEG', quality=95)
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = 'image/jpeg'
        elif input_type == 'url':
            # Download and convert URL image
            img = download_image_from_url(image_input)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = 'image/jpeg'
        else:  # file_path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            
            # Use original image file
            with open(image_input, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine MIME type based on file extension
            ext = os.path.splitext(image_input)[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg', 
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')
    
    return f"data:{mime_type};base64,{encoded_string}"

def check_vllm_server_status(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Check vLLM server status and retrieve server information.
    
    Args:
        api_url: Base URL of the vLLM server
        
    Returns:
        Dictionary containing server status and information
    """
    result = {
        "server_running": False,
        "models": [],
        "server_info": {},
        "error": None
    }
    
    try:
        # Check if server is running with health endpoint
        health_url = f"{api_url}/health"
        health_response = requests.get(health_url, timeout=5)
        
        if health_response.status_code == 200:
            result["server_running"] = True
            print(f"✅ vLLM server is running at {api_url}")
        else:
            result["error"] = f"Server returned status code: {health_response.status_code}"
            return result
            
    except requests.exceptions.RequestException as e:
        result["error"] = f"Cannot connect to server: {str(e)}"
        return result
    
    try:
        # Get model information
        models_url = f"{api_url}/v1/models"
        models_response = requests.get(models_url, timeout=10)
        
        if models_response.status_code == 200:
            models_data = models_response.json()
            result["models"] = models_data.get("data", [])
            
            print(f"\n📋 Available Models:")
            for model in result["models"]:
                print(f"  - Model ID: {model.get('id', 'Unknown')}")
                print(f"    Created: {model.get('created', 'Unknown')}")
                print(f"    Object: {model.get('object', 'Unknown')}")
                print(f"    Owned by: {model.get('owned_by', 'Unknown')}")
                
                # Try to get more detailed model info
                if 'id' in model:
                    model_id = model['id']
                    print(f"    Model Path: {model_id}")
                    
                    # Check for quantization indicators in model name
                    quantization_type = detect_quantization_type(model_id)
                    if quantization_type:
                        print(f"    Quantization: {quantization_type}")
                    else:
                        print(f"    Quantization: None detected (likely FP16/BF16)")
                print()
        else:
            result["error"] = f"Failed to get models: {models_response.status_code}"
            
    except requests.exceptions.RequestException as e:
        result["error"] = f"Failed to get model info: {str(e)}"
    
    return result

def detect_quantization_type(model_name: str) -> Optional[str]:
    """
    Detect quantization type from model name/path.
    
    Args:
        model_name: Model name or path
        
    Returns:
        Quantization type string or None
    """
    model_lower = model_name.lower()
    
    # Common quantization patterns
    if 'awq' in model_lower:
        return "AWQ (4-bit)"
    elif 'gptq' in model_lower:
        if 'int4' in model_lower or '4bit' in model_lower:
            return "GPTQ (4-bit)"
        elif 'int8' in model_lower or '8bit' in model_lower:
            return "GPTQ (8-bit)"
        else:
            return "GPTQ"
    elif 'gguf' in model_lower or 'ggml' in model_lower:
        return "GGUF/GGML"
    elif 'int8' in model_lower or '8bit' in model_lower:
        return "INT8"
    elif 'int4' in model_lower or '4bit' in model_lower:
        return "INT4"
    elif 'fp8' in model_lower:
        return "FP8"
    elif 'bitsandbytes' in model_lower or 'bnb' in model_lower:
        return "BitsAndBytes"
    
    return None

def get_server_config_info(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Try to get server configuration information.
    Note: This may not work with all vLLM versions as the endpoint might not be available.
    
    Args:
        api_url: Base URL of the vLLM server
        
    Returns:
        Dictionary containing configuration information
    """
    config_info = {
        "max_model_len": "Unknown",
        "served_model_name": "Unknown",
        "engine_config": {},
        "error": None
    }
    
    try:
        # Try to get server stats/info (this endpoint may vary by vLLM version)
        endpoints_to_try = [
            f"{api_url}/stats",
            f"{api_url}/v1/stats", 
            f"{api_url}/metrics",
            f"{api_url}/info"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"📊 Server Configuration (from {endpoint}):")
                    
                    # Look for common configuration fields
                    if 'max_model_len' in data:
                        config_info["max_model_len"] = data['max_model_len']
                        print(f"  Max Model Length: {data['max_model_len']:,} tokens")
                    
                    if 'served_model_name' in data:
                        config_info["served_model_name"] = data['served_model_name']
                        print(f"  Served Model: {data['served_model_name']}")
                    
                    # Print other interesting fields
                    interesting_fields = [
                        'num_gpus', 'gpu_memory_utilization', 'max_num_seqs',
                        'max_num_batched_tokens', 'tensor_parallel_size',
                        'pipeline_parallel_size', 'quantization'
                    ]
                    
                    for field in interesting_fields:
                        if field in data:
                            print(f"  {field.replace('_', ' ').title()}: {data[field]}")
                    
                    config_info["engine_config"] = data
                    break
                    
            except requests.exceptions.RequestException:
                continue
        
        if not config_info["engine_config"]:
            print("⚠️  Could not retrieve detailed server configuration")
            print("   This is normal - configuration endpoints vary by vLLM version")
            
    except Exception as e:
        config_info["error"] = str(e)
    
    return config_info

def estimate_context_usage_from_server(api_url: str = "http://localhost:8000") -> None:
    """
    Make a simple request to estimate context usage capabilities.
    
    Args:
        api_url: Base URL of the vLLM server
    """
    try:
        print("\n🧪 Testing Context Usage Estimation:")
        
        # Make a simple text-only request to see response
        test_payload = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",  # This will be auto-detected
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What is your maximum context length?"}
                    ]
                }
            ],
            "max_tokens": 100,
            "stream": False
        }
        
        response = requests.post(f"{api_url}/v1/chat/completions", 
                               json=test_payload, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'usage' in data:
                usage = data['usage']
                print(f"  Prompt tokens: {usage.get('prompt_tokens', 'Unknown')}")
                print(f"  Completion tokens: {usage.get('completion_tokens', 'Unknown')}")
                print(f"  Total tokens: {usage.get('total_tokens', 'Unknown')}")
                
                # Try to extract context info from response
                if 'choices' in data and data['choices']:
                    content = data['choices'][0].get('message', {}).get('content', '')
                    if any(keyword in content.lower() for keyword in ['32768', '32k', 'context', 'length']):
                        print(f"  Model response about context: {content[:200]}...")
            else:
                print("  No usage information returned")
        else:
            print(f"  Test request failed: {response.status_code}")
            
    except Exception as e:
        print(f"  Context test failed: {str(e)}")

def comprehensive_server_check(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Perform a comprehensive check of the vLLM server.
    
    Args:
        api_url: Base URL of the vLLM server
        
    Returns:
        Dictionary containing all server information
    """
    print("🔍 Comprehensive vLLM Server Check")
    print("=" * 50)
    
    # Check server status and models
    server_status = check_vllm_server_status(api_url)
    
    if not server_status["server_running"]:
        print(f"❌ Server check failed: {server_status['error']}")
        return server_status
    
    # Get configuration info
    config_info = get_server_config_info(api_url)
    
    # Test context usage
    estimate_context_usage_from_server(api_url)
    
    # Combine results
    result = {
        **server_status,
        "config": config_info,
        "api_url": api_url,
        "check_timestamp": time.time()
    }
    
    print("\n✅ Server check completed!")
    return result

def calculate_token_speed(start_time: float, end_time: float, total_tokens: int, 
                         prompt_tokens: int = 0, completion_tokens: int = 0) -> Dict[str, float]:
    """
    Calculate token generation speed metrics.
    
    Args:
        start_time: Request start timestamp
        end_time: Request end timestamp  
        total_tokens: Total tokens used
        prompt_tokens: Prompt tokens (optional)
        completion_tokens: Completion tokens (optional)
        
    Returns:
        Dictionary with speed metrics
    """
    duration = end_time - start_time
    
    if duration <= 0:
        return {"error": "Invalid duration"}
    
    metrics = {
        "duration_seconds": duration,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }
    
    if completion_tokens > 0:
        # Calculate generation speed (excluding prompt processing time)
        # Assume prompt processing takes ~10% of total time for estimation
        generation_time = duration * 0.9
        metrics["generation_tokens_per_second"] = completion_tokens / generation_time
    
    return metrics

def process_single_image(image_input: Union[str, Image.Image], prompt: str = "Describe this image in detail. What objects do you see?",
                        api_url: str = "http://localhost:8000/v1/chat/completions",
                        max_tokens: int = 1024, resize_for_context: bool = True) -> Dict[str, Any]:
    """
    Process a single image with timing and performance metrics.
    
    Args:
        image_input: Image input (URL, file path, or PIL Image)
        prompt: Text prompt for the model
        api_url: vLLM server API endpoint
        max_tokens: Maximum tokens to generate
        resize_for_context: Whether to resize image for context limits
        
    Returns:
        Dictionary containing response and performance metrics
    """
    result = {
        "image_path": str(image_input) if not isinstance(image_input, Image.Image) else "PIL_Image",
        "success": False,
        "response": None,
        "performance": {},
        "error": None
    }
    
    try:
        # Start timing
        start_time = time.time()
        
        # Process image
        input_type = detect_image_input_type(image_input)
        if input_type == 'pil_image':
            print(f"📸 Processing PIL Image")
        elif input_type == 'url':
            print(f"📸 Processing image from URL: {image_input}")
        else:
            print(f"📸 Processing image: {os.path.basename(image_input)}")
            
        image_data_url = encode_image_to_base64(image_input, resize_for_context=resize_for_context)
        
        # Get image info for logging
        if input_type == 'pil_image':
            img = image_input
            width, height = img.size
        elif input_type == 'url':
            img = download_image_from_url(image_input)
            width, height = img.size
        else:
            with Image.open(image_input) as img:
                width, height = img.size
                
        total_pixels = width * height
        estimated_tokens = total_pixels // (28 * 28)
            
        print(f"   Dimensions: {width}x{height} ({total_pixels:,} pixels)")
        print(f"   Estimated visual tokens: {estimated_tokens:,}")
        
        # Create payload
        payload = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Send request
        request_start = time.time()
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        request_end = time.time()
        
        if response.status_code == 200:
            response_data = response.json()
            result["success"] = True
            result["response"] = response_data
            
            # Calculate performance metrics
            if 'usage' in response_data:
                usage = response_data['usage']
                total_tokens = usage.get('total_tokens', 0)
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                
                speed_metrics = calculate_token_speed(
                    request_start, request_end, total_tokens, prompt_tokens, completion_tokens
                )
                
                result["performance"] = {
                    **speed_metrics,
                    "image_processing_time": request_start - start_time,
                    "total_processing_time": request_end - start_time,
                    "estimated_visual_tokens": estimated_tokens,
                    "image_dimensions": f"{width}x{height}",
                    "image_pixels": total_pixels
                }
                
                print(f"   ⚡ Speed: {speed_metrics['tokens_per_second']:.1f} tokens/sec")
                if 'generation_tokens_per_second' in speed_metrics:
                    print(f"   🚀 Generation: {speed_metrics['generation_tokens_per_second']:.1f} tokens/sec")
            
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        result["error"] = str(e)
    
    return result

def process_batch_images(image_paths: List[str], prompt: str = "Describe this image in detail. What objects do you see?",
                        api_url: str = "http://localhost:8000/v1/chat/completions",
                        max_tokens: int = 1024, resize_for_context: bool = True,
                        concurrent: bool = False) -> Dict[str, Any]:
    """
    Process multiple images with performance analysis.
    
    Args:
        image_paths: List of image file paths
        prompt: Text prompt for the model
        api_url: vLLM server API endpoint
        max_tokens: Maximum tokens to generate per image
        resize_for_context: Whether to resize images for context limits
        concurrent: Whether to process images concurrently (not implemented yet)
        
    Returns:
        Dictionary containing batch results and performance metrics
    """
    print(f"\n🔄 Processing batch of {len(image_paths)} images")
    print("=" * 60)
    
    batch_start = time.time()
    results = []
    performance_metrics = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
        
        result = process_single_image(
            image_path, prompt, api_url, max_tokens, resize_for_context
        )
        results.append(result)
        
        if result["success"] and result["performance"]:
            performance_metrics.append(result["performance"])
    
    batch_end = time.time()
    batch_duration = batch_end - batch_start
    
    # Calculate batch statistics
    batch_stats = calculate_batch_statistics(performance_metrics, batch_duration)
    
    print(f"\n📊 Batch Processing Summary:")
    print("=" * 40)
    print(f"Total images: {len(image_paths)}")
    print(f"Successful: {len([r for r in results if r['success']])}")
    print(f"Failed: {len([r for r in results if not r['success']])}")
    print(f"Total time: {batch_duration:.2f} seconds")
    
    if batch_stats:
        print(f"Average speed: {batch_stats['avg_tokens_per_second']:.1f} tokens/sec")
        print(f"Total tokens: {batch_stats['total_tokens']:,}")
        print(f"Throughput: {batch_stats['images_per_minute']:.1f} images/min")
    
    return {
        "results": results,
        "batch_statistics": batch_stats,
        "total_duration": batch_duration,
        "success_count": len([r for r in results if r['success']]),
        "error_count": len([r for r in results if not r['success']])
    }

def calculate_batch_statistics(performance_metrics: List[Dict[str, Any]], 
                              batch_duration: float) -> Dict[str, Any]:
    """
    Calculate statistics for batch processing performance.
    
    Args:
        performance_metrics: List of performance metrics from individual images
        batch_duration: Total batch processing time
        
    Returns:
        Dictionary with batch statistics
    """
    if not performance_metrics:
        return {}
    
    # Extract metrics
    speeds = [m['tokens_per_second'] for m in performance_metrics if 'tokens_per_second' in m]
    total_tokens = sum(m['total_tokens'] for m in performance_metrics if 'total_tokens' in m)
    durations = [m['duration_seconds'] for m in performance_metrics if 'duration_seconds' in m]
    
    stats = {
        "total_tokens": total_tokens,
        "total_images": len(performance_metrics),
        "batch_duration": batch_duration,
        "images_per_minute": (len(performance_metrics) / batch_duration) * 60,
        "avg_tokens_per_second": statistics.mean(speeds) if speeds else 0,
        "median_tokens_per_second": statistics.median(speeds) if speeds else 0,
        "min_tokens_per_second": min(speeds) if speeds else 0,
        "max_tokens_per_second": max(speeds) if speeds else 0,
        "avg_processing_time": statistics.mean(durations) if durations else 0,
        "total_throughput_tokens_per_second": total_tokens / batch_duration if batch_duration > 0 else 0
    }
    
    # Add standard deviation if we have enough data points
    if len(speeds) > 1:
        stats["std_tokens_per_second"] = statistics.stdev(speeds)
    
    return stats

# Define the server endpoint
api_url = "http://localhost:8000/v1/chat/completions"

if __name__ == "__main__":
    # First, perform comprehensive server check
    print("🚀 Starting vLLM Server Check and Image Processing Test")
    print("=" * 60)
    
    # Check server status and configuration
    server_info = comprehensive_server_check("http://localhost:8000")
    
    if not server_info.get("server_running", False):
        print("\n❌ Cannot proceed with image test - server is not running")
        print("\nTo start the vLLM server, run:")
        print("vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 32768")
        exit(1)
    
    print("\n" + "="*60)
    print("🖼️  Testing Image Processing with Performance Metrics")
    print("="*60)
    
    # Define test images (you can modify this list)
    test_images = [
        "VisionLangAnnotateModels/sampledata/bus.jpg",
        "bus.jpg",  # Alternative path
        "test_image.jpg"  # Add more images as needed
    ]
    
    # Filter existing images
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print(f"❌ No test images found! Please make sure at least one of these images exists:")
        for img in test_images:
            print(f"   - {img}")
        exit(1)
    
    print(f"\n📋 Found {len(existing_images)} test image(s): {', '.join([os.path.basename(img) for img in existing_images])}")
    
    # Decide processing mode based on number of images
    if len(existing_images) == 1:
        print("\n🔄 Single Image Processing Mode")
        print("=" * 50)
        
        # Process single image with detailed metrics
        result = process_single_image(
            existing_images[0], 
            prompt="Describe this image in detail. What objects do you see?",
            api_url=api_url,
            max_tokens=1024,
            resize_for_context=True
        )
        
        if result["success"]:
            print("\n" + "="*50)
            print("🤖 MODEL RESPONSE:")
            print("="*50)
            if result["response"] and isinstance(result["response"], dict) and "choices" in result["response"]:
                print(result["response"]["choices"][0]["message"]["content"])
            print("="*50)
            
            # Print detailed performance metrics
            if result["performance"]:
                perf = result["performance"]
                print(f"\n📊 Performance Metrics:")
                print(f"   Image processing: {perf['image_processing_time']:.3f}s")
                print(f"   API request: {perf['duration_seconds']:.3f}s")
                print(f"   Total time: {perf['total_processing_time']:.3f}s")
                print(f"   Speed: {perf['tokens_per_second']:.1f} tokens/sec")
                if 'generation_tokens_per_second' in perf:
                    print(f"   Generation speed: {perf['generation_tokens_per_second']:.1f} tokens/sec")
                
                print(f"\n📊 Token Usage:")
                print(f"   Prompt tokens: {perf['prompt_tokens']:,}")
                print(f"   Completion tokens: {perf['completion_tokens']:,}")
                print(f"   Total tokens: {perf['total_tokens']:,}")
                print(f"   Visual tokens (estimated): {perf['estimated_visual_tokens']:,}")
                
                # Context usage
                max_len = server_info.get("config", {}).get("max_model_len", "Unknown")
                if max_len != "Unknown" and isinstance(max_len, int):
                    usage_percent = (perf['total_tokens'] / max_len) * 100
                    print(f"   Context usage: {usage_percent:.1f}% of {max_len:,} max tokens")
            
            print(f"\n✅ Image processing completed successfully!")
        else:
            print(f"❌ Processing failed: {result['error']}")
    
    else:
        print("\n🔄 Batch Processing Mode")
        print("=" * 50)
        
        # Process multiple images with batch analysis
        batch_result = process_batch_images(
            existing_images,
            prompt="Describe this image in detail. What objects do you see?",
            api_url=api_url,
            max_tokens=1024,
            resize_for_context=True
        )
        
        # Show individual results
        print(f"\n📋 Individual Results:")
        print("=" * 50)
        for i, result in enumerate(batch_result["results"], 1):
            if result["success"]:
                print(f"\n[{i}] {os.path.basename(result['image_path'])}:")
                if result["response"] and isinstance(result["response"], dict) and "choices" in result["response"]:
                    response_text = result["response"]["choices"][0]["message"]["content"]
                    # Truncate long responses for batch display
                    if len(response_text) > 200:
                        response_text = response_text[:200] + "..."
                    print(f"   Response: {response_text}")
                else:
                    print(f"   Response: No valid response received")
                
                if result["performance"]:
                    perf = result["performance"]
                    print(f"   Speed: {perf['tokens_per_second']:.1f} tokens/sec")
                    print(f"   Tokens: {perf['total_tokens']:,}")
            else:
                print(f"\n[{i}] {os.path.basename(result['image_path'])}: ❌ {result['error']}")
        
        # Show batch statistics
        if batch_result["batch_statistics"]:
            stats = batch_result["batch_statistics"]
            print(f"\n📈 Batch Performance Analysis:")
            print("=" * 40)
            print(f"Processing efficiency: {stats['images_per_minute']:.1f} images/min")
            print(f"Average speed: {stats['avg_tokens_per_second']:.1f} ± {stats.get('std_tokens_per_second', 0):.1f} tokens/sec")
            print(f"Speed range: {stats['min_tokens_per_second']:.1f} - {stats['max_tokens_per_second']:.1f} tokens/sec")
            print(f"Total throughput: {stats['total_throughput_tokens_per_second']:.1f} tokens/sec")
            
            # Context usage for batch
            max_len = server_info.get("config", {}).get("max_model_len", "Unknown")
            if max_len != "Unknown" and isinstance(max_len, int):
                avg_usage_percent = (stats['total_tokens'] / len(existing_images) / max_len) * 100
                print(f"Average context usage: {avg_usage_percent:.1f}% per image")
    
    print(f"\n🎉 Processing completed successfully!")
    if server_info.get("models"):
        model_name = server_info["models"][0].get("id", "Unknown model")
        print(f"   Server: {model_name}")
        quantization = detect_quantization_type(model_name)
        print(f"   Quantization: {quantization or 'None detected'}")
    max_len = server_info.get("config", {}).get("max_model_len", "Unknown")
    if max_len != "Unknown":
        print(f"   Max context: {max_len:,} tokens")

#three cases
# # URL
# process_single_image("https://example.com/image.jpg", prompt="Describe this")

# # File path  
# process_single_image("/path/to/image.jpg", prompt="Describe this")

# # PIL Image
# from PIL import Image
# img = Image.open("image.jpg")
# process_single_image(img, prompt="Describe this")