#!/usr/bin/env python3
"""
Enhanced VLM Utility Class

A comprehensive utility class that provides a unified interface for vision-language model operations
across multiple backends:
1. vLLM API Mode: Uses existing vLLM server via HTTP API
2. vLLM Package Mode: Uses vLLM Python package directly
3. Ollama Mode: Uses Ollama API for local model inference

This class provides consistent functionality regardless of the underlying implementation.
"""

import requests
import json
import base64
import os
from PIL import Image
import io
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Literal
import statistics
import urllib.request
import urllib.parse
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

class VLMBackend(Enum):
    """Enumeration for VLM backend types."""
    VLLM_API = "vllm_api"
    VLLM_PACKAGE = "vllm_package"
    OLLAMA = "ollama"

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

class VLLMAPIBackend(VLMBackendInterface):
    """vLLM API backend implementation."""
    
    def __init__(self, api_url: str = "http://localhost:8000", model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.is_initialized = False
    
    def initialize(self, **kwargs) -> bool:
        """Initialize vLLM API backend."""
        try:
            health_url = f"{self.api_url}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Connected to vLLM server at {self.api_url}")
                
                # Fetch actual model name from server
                try:
                    models_url = f"{self.api_url}/v1/models"
                    models_response = requests.get(models_url, timeout=10)
                    
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        models = models_data.get("data", [])
                        
                        if models:
                            server_model_name = models[0].get("id", self.model_name)
                            if server_model_name != self.model_name:
                                logger.info(f"🔄 Updated model name from '{self.model_name}' to '{server_model_name}'")
                                self.model_name = server_model_name
                            else:
                                logger.info(f"✅ Using model: {self.model_name}")
                        else:
                            logger.warning(f"⚠️  No models found on server, using default: {self.model_name}")
                    else:
                        logger.warning(f"⚠️  Could not fetch models from server, using default: {self.model_name}")
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"⚠️  Could not fetch model info: {str(e)}, using default: {self.model_name}")
                
                self.is_initialized = True
                return True
            else:
                logger.error(f"Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to vLLM server: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Check if vLLM API backend is available."""
        return self.is_initialized
    
    def process_image(self, image_input: Union[str, Image.Image], prompt: str, **kwargs) -> Dict[str, Any]:
        """Process image using vLLM API."""
        if not self.is_available():
            return {"success": False, "error": "vLLM API backend not initialized"}
        
        result = {
            "success": False,
            "response": None,
            "performance": {},
            "error": None,
            "backend": "vllm_api"
        }
        
        try:
            start_time = time.time()
            
            # Convert image to base64
            image_data_url = self._encode_image_to_base64(image_input, kwargs.get('resize_for_context', True))
            
            # Create payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}}
                        ]
                    }
                ],
                "max_tokens": kwargs.get('max_tokens', 1024),
                "stream": False
            }
            
            # Add additional parameters
            for key in ['temperature', 'top_p', 'top_k']:
                if key in kwargs:
                    payload[key] = kwargs[key]
            
            # Make API request
            response = requests.post(f"{self.api_url}/v1/chat/completions", 
                                   json=payload, 
                                   timeout=120)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if 'choices' in data and data['choices']:
                    result["response"] = data['choices'][0]['message']['content']
                    result["success"] = True
                
                if 'usage' in data:
                    usage = data['usage']
                    result["performance"] = {
                        "duration_seconds": end_time - start_time,
                        "prompt_tokens": usage.get('prompt_tokens', 0),
                        "completion_tokens": usage.get('completion_tokens', 0),
                        "total_tokens": usage.get('total_tokens', 0)
                    }
                    
                    if result["performance"]["total_tokens"] > 0:
                        result["performance"]["tokens_per_second"] = (
                            result["performance"]["total_tokens"] / result["performance"]["duration_seconds"]
                        )
            else:
                result["error"] = f"API request failed: {response.status_code} - {response.text}"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM API model information."""
        if not self.is_available():
            return {"error": "Backend not available"}
        
        try:
            models_url = f"{self.api_url}/v1/models"
            response = requests.get(models_url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get model info: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _encode_image_to_base64(self, image_input: Union[str, Image.Image], resize_for_context: bool = True) -> str:
        """Convert image to base64 data URL."""
        # This is a simplified version - in practice, you'd use the full implementation from the original class
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # Download from URL
                with urllib.request.urlopen(image_input) as response:
                    image_data = response.read()
                    img = Image.open(io.BytesIO(image_data))
            else:
                # Load from file
                img = Image.open(image_input)
        else:
            img = image_input
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if requested (simplified logic)
        if resize_for_context:
            max_pixels = 1280 * 28 * 28
            width, height = img.size
            if width * height > max_pixels:
                scale_factor = (max_pixels / (width * height)) ** 0.5
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{encoded_string}"

class VLLMPackageBackend(VLMBackendInterface):
    """vLLM Package backend implementation."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name = model_name
        self.llm = None
        self.is_initialized = False
    
    def initialize(self, **kwargs) -> bool:
        """Initialize vLLM package backend."""
        if not VLLM_AVAILABLE:
            logger.error("vLLM package is not available. Install it with: pip install vllm")
            return False
        
        default_kwargs = {
            "trust_remote_code": True,
            "max_model_len": 1024,
            "limit_mm_per_prompt": {"image": 1},
            "enforce_eager": True,
            "gpu_memory_utilization": 0.25,
            "disable_log_stats": True,
            "disable_custom_all_reduce": True
        }
        default_kwargs.update(kwargs)
        
        try:
            self.llm = LLM(model=self.model_name, **default_kwargs)
            logger.info(f"✅ vLLM package mode initialized with model: {self.model_name}")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vLLM package mode: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Check if vLLM package backend is available."""
        return self.is_initialized and self.llm is not None
    
    def process_image(self, image_input: Union[str, Image.Image], prompt: str, **kwargs) -> Dict[str, Any]:
        """Process image using vLLM package."""
        if not self.is_available():
            return {"success": False, "error": "vLLM package backend not initialized"}
        
        result = {
            "success": False,
            "response": None,
            "performance": {},
            "error": None,
            "backend": "vllm_package"
        }
        
        try:
            start_time = time.time()
            
            # Process image
            if isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    with urllib.request.urlopen(image_input) as response:
                        image_data = response.read()
                        img = Image.open(io.BytesIO(image_data))
                else:
                    img = Image.open(image_input)
            else:
                img = image_input
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=kwargs.get('max_tokens', 1024),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9)
            )
            
            # Format prompt for Qwen2.5-VL
            formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            # Create prompt input with multi_modal_data
            prompt_input = {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": img}
            }
            
            # Generate response
            outputs = self.llm.generate(prompt_input, sampling_params=sampling_params)
            
            end_time = time.time()
            
            if outputs:
                output = outputs[0]
                result["response"] = output.outputs[0].text
                result["success"] = True
                
                result["performance"] = {
                    "duration_seconds": end_time - start_time,
                    "prompt_tokens": len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0,
                    "completion_tokens": len(output.outputs[0].token_ids) if output.outputs else 0,
                    "total_tokens": 0
                }
                
                result["performance"]["total_tokens"] = (
                    result["performance"]["prompt_tokens"] + result["performance"]["completion_tokens"]
                )
                
                if result["performance"]["total_tokens"] > 0:
                    result["performance"]["tokens_per_second"] = (
                        result["performance"]["total_tokens"] / result["performance"]["duration_seconds"]
                    )
            else:
                result["error"] = "No output generated"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM package model information."""
        if not self.is_available():
            return {"error": "Backend not available"}
        
        return {
            "model_name": self.model_name,
            "backend": "vllm_package",
            "available": True
        }

class OllamaBackend(VLMBackendInterface):
    """Ollama backend implementation."""
    
    def __init__(self, api_url: str = "http://localhost:11434", model_name: str = "llava"):
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.is_initialized = False
    
    def initialize(self, **kwargs) -> bool:
        """Initialize Ollama backend."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                if self.model_name in available_models:
                    logger.info(f"✅ Connected to Ollama at {self.api_url} with model: {self.model_name}")
                    self.is_initialized = True
                    return True
                else:
                    logger.warning(f"⚠️  Model {self.model_name} not found. Available models: {available_models}")
                    if available_models:
                        # Use first available vision model
                        vision_models = [m for m in available_models if 'llava' in m.lower() or 'vision' in m.lower()]
                        if vision_models:
                            self.model_name = vision_models[0]
                            logger.info(f"🔄 Using available vision model: {self.model_name}")
                            self.is_initialized = True
                            return True
                    return False
            else:
                logger.error(f"Ollama server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama server: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Check if Ollama backend is available."""
        return self.is_initialized
    
    def process_image(self, image_input: Union[str, Image.Image], prompt: str, **kwargs) -> Dict[str, Any]:
        """Process image using Ollama."""
        if not self.is_available():
            return {"success": False, "error": "Ollama backend not initialized"}
        
        result = {
            "success": False,
            "response": None,
            "performance": {},
            "error": None,
            "backend": "ollama"
        }
        
        try:
            start_time = time.time()
            
            # Convert image to base64
            image_base64 = self._encode_image_to_base64(image_input)
            
            # Create payload for Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "num_predict": kwargs.get('max_tokens', 1024),
                    "temperature": kwargs.get('temperature', 0.7),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 40)
                }
            }
            
            # Make API request
            response = requests.post(f"{self.api_url}/api/generate", 
                                   json=payload, 
                                   timeout=120)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if 'response' in data:
                    result["response"] = data['response']
                    result["success"] = True
                
                # Extract performance metrics if available
                result["performance"] = {
                    "duration_seconds": end_time - start_time,
                    "prompt_eval_count": data.get('prompt_eval_count', 0),
                    "eval_count": data.get('eval_count', 0),
                    "total_duration": data.get('total_duration', 0),
                    "load_duration": data.get('load_duration', 0),
                    "prompt_eval_duration": data.get('prompt_eval_duration', 0),
                    "eval_duration": data.get('eval_duration', 0)
                }
                
                # Calculate tokens per second if eval_duration is available
                if data.get('eval_duration', 0) > 0 and data.get('eval_count', 0) > 0:
                    result["performance"]["tokens_per_second"] = (
                        data['eval_count'] / (data['eval_duration'] / 1e9)  # Convert nanoseconds to seconds
                    )
            else:
                result["error"] = f"Ollama API request failed: {response.status_code} - {response.text}"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        if not self.is_available():
            return {"error": "Backend not available"}
        
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                current_model = next((m for m in models if m['name'] == self.model_name), None)
                
                return {
                    "model_name": self.model_name,
                    "backend": "ollama",
                    "available": True,
                    "model_info": current_model,
                    "all_models": [m['name'] for m in models]
                }
            else:
                return {"error": f"Failed to get model info: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _encode_image_to_base64(self, image_input: Union[str, Image.Image]) -> str:
        """Convert image to base64 string (without data URL prefix for Ollama)."""
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # Download from URL
                with urllib.request.urlopen(image_input) as response:
                    image_data = response.read()
                    return base64.b64encode(image_data).decode('utf-8')
            else:
                # Load from file
                with open(image_input, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # PIL Image
            buffer = io.BytesIO()
            if image_input.mode != 'RGB':
                image_input = image_input.convert('RGB')
            image_input.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

class VLMUtils:
    """
    Enhanced VLM Utility Class with multiple backend support.
    
    Supports three backends:
    1. vLLM API: Communicates with a running vLLM server via HTTP API
    2. vLLM Package: Uses vLLM Python package directly for inference
    3. Ollama: Uses Ollama API for local model inference
    """
    
    def __init__(self, 
                 backend: Union[VLMBackend, str] = VLMBackend.VLLM_API,
                 **kwargs):
        """
        Initialize the VLM utility with specified backend.
        
        Args:
            backend: Backend type to use
            **kwargs: Backend-specific configuration parameters
        """
        if isinstance(backend, str):
            backend = VLMBackend(backend)
        
        self.backend_type = backend
        self.backend = None
        self.fallback_backends = []
        
        # Initialize primary backend
        self._initialize_backend(backend, **kwargs)
        
        # Setup fallback backends if primary fails
        self._setup_fallbacks(**kwargs)
    
    def _initialize_backend(self, backend_type: VLMBackend, **kwargs) -> bool:
        """Initialize the specified backend."""
        try:
            if backend_type == VLMBackend.VLLM_API:
                self.backend = VLLMAPIBackend(
                    api_url=kwargs.get('api_url', 'http://localhost:8000'),
                    model_name=kwargs.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
                )
            elif backend_type == VLMBackend.VLLM_PACKAGE:
                self.backend = VLLMPackageBackend(
                    model_name=kwargs.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
                )
            elif backend_type == VLMBackend.OLLAMA:
                self.backend = OllamaBackend(
                    api_url=kwargs.get('ollama_url', 'http://localhost:11434'),
                    model_name=kwargs.get('ollama_model', 'llava')
                )
            
            if self.backend and self.backend.initialize(**kwargs):
                logger.info(f"✅ Successfully initialized {backend_type.value} backend")
                return True
            else:
                logger.error(f"❌ Failed to initialize {backend_type.value} backend")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing {backend_type.value} backend: {str(e)}")
            return False
    
    def _setup_fallbacks(self, **kwargs):
        """Setup fallback backends."""
        fallback_order = [VLMBackend.VLLM_API, VLMBackend.OLLAMA, VLMBackend.VLLM_PACKAGE]
        
        for fallback_type in fallback_order:
            if fallback_type != self.backend_type:
                try:
                    if fallback_type == VLMBackend.VLLM_API:
                        fallback = VLLMAPIBackend(
                            api_url=kwargs.get('api_url', 'http://localhost:8000'),
                            model_name=kwargs.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
                        )
                        # Only pass relevant kwargs for vLLM API
                        init_kwargs = {}
                    elif fallback_type == VLMBackend.OLLAMA:
                        fallback = OllamaBackend(
                            api_url=kwargs.get('ollama_url', 'http://localhost:11434'),
                            model_name=kwargs.get('ollama_model', 'llava')
                        )
                        # Only pass relevant kwargs for Ollama
                        init_kwargs = {}
                    elif fallback_type == VLMBackend.VLLM_PACKAGE:
                        fallback = VLLMPackageBackend(
                            model_name=kwargs.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
                        )
                        # Pass vLLM package specific kwargs
                        init_kwargs = {k: v for k, v in kwargs.items() 
                                     if k in ['trust_remote_code', 'max_model_len', 'limit_mm_per_prompt', 
                                            'enforce_eager', 'gpu_memory_utilization', 'disable_log_stats', 
                                            'disable_custom_all_reduce']}
                    
                    if fallback.initialize(**init_kwargs):
                        self.fallback_backends.append(fallback)
                        logger.info(f"✅ Fallback backend {fallback_type.value} ready")
                except Exception as e:
                    logger.debug(f"Fallback backend {fallback_type.value} not available: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if any backend is available."""
        return (self.backend and self.backend.is_available()) or len(self.fallback_backends) > 0
    
    def process_image(self, 
                     image_input: Union[str, Image.Image], 
                     prompt: str = "Describe this image in detail. What objects do you see?",
                     **kwargs) -> Dict[str, Any]:
        """
        Process a single image using the available backend.
        
        Args:
            image_input: Image input (URL, file path, or PIL Image)
            prompt: Text prompt for the model
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            Dictionary containing response and performance metrics
        """
        # Try primary backend first
        if self.backend and self.backend.is_available():
            result = self.backend.process_image(image_input, prompt, **kwargs)
            if result.get('success', False):
                return result
            else:
                logger.warning(f"Primary backend failed: {result.get('error', 'Unknown error')}")
        
        # Try fallback backends
        for fallback in self.fallback_backends:
            if fallback.is_available():
                logger.info(f"Trying fallback backend...")
                result = fallback.process_image(image_input, prompt, **kwargs)
                if result.get('success', False):
                    return result
                else:
                    logger.warning(f"Fallback backend failed: {result.get('error', 'Unknown error')}")
        
        return {
            "success": False,
            "error": "All backends failed or unavailable",
            "backend": "none"
        }
    
    def process_multiple_images(self, 
                               image_inputs: List[Union[str, Image.Image]], 
                               prompts: Union[str, List[str]] = "Describe this image in detail. What objects do you see?",
                               **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple images.
        
        Args:
            image_inputs: List of image inputs
            prompts: Single prompt or list of prompts (one per image)
            **kwargs: Additional parameters
            
        Returns:
            List of result dictionaries
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(image_inputs)
        elif len(prompts) != len(image_inputs):
            raise ValueError("Number of prompts must match number of images")
        
        results = []
        for image_input, prompt in zip(image_inputs, prompts):
            result = self.process_image(image_input, prompt, **kwargs)
            results.append(result)
        
        return results
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        info = {
            "primary_backend": self.backend_type.value if self.backend_type else "none",
            "primary_available": self.backend.is_available() if self.backend else False,
            "fallback_backends": [],
            "total_available": 0
        }
        
        if self.backend and self.backend.is_available():
            info["primary_model_info"] = self.backend.get_model_info()
            info["total_available"] += 1
        
        for fallback in self.fallback_backends:
            if fallback.is_available():
                backend_name = fallback.__class__.__name__.replace('Backend', '').lower()
                info["fallback_backends"].append({
                    "backend": backend_name,
                    "model_info": fallback.get_model_info()
                })
                info["total_available"] += 1
        
        return info
    
    def get_performance_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate performance summary from multiple results.
        
        Args:
            results: List of result dictionaries from process_image calls
            
        Returns:
            Dictionary containing performance statistics
        """
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        durations = [r['performance'].get('duration_seconds', 0) for r in successful_results if 'performance' in r]
        tokens_per_second = [r['performance'].get('tokens_per_second', 0) for r in successful_results if 'performance' in r]
        
        summary = {
            "total_images": len(results),
            "successful_images": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "backends_used": list(set(r.get('backend', 'unknown') for r in successful_results))
        }
        
        if durations:
            summary["performance"] = {
                "avg_duration_seconds": statistics.mean(durations),
                "min_duration_seconds": min(durations),
                "max_duration_seconds": max(durations),
                "median_duration_seconds": statistics.median(durations)
            }
        
        if tokens_per_second:
            summary["performance"]["avg_tokens_per_second"] = statistics.mean([tps for tps in tokens_per_second if tps > 0])
            summary["performance"]["median_tokens_per_second"] = statistics.median([tps for tps in tokens_per_second if tps > 0])
        
        return summary


def example_usage():
    """Example usage of the VLMUtils class."""
    
    print("=== VLMUtils Multi-Backend Example ===")
    
    # Example 1: Auto-detect and use best available backend
    print("\n1. Auto-detection mode:")
    try:
        vlm = VLMUtils()  # Will try vLLM API first, then fallbacks
        
        backend_info = vlm.get_backend_info()
        print(f"Backend info: {backend_info}")
        
        if vlm.is_available():
            result = vlm.process_image(
                "VisionLangAnnotateModels/sampledata/sjsupeople.jpg", 
                "What do you see in this image?"
            )
            print(f"Result: {result}")
        else:
            print("No backends available")
            
    except Exception as e:
        print(f"Auto-detection error: {e}")
    
    # Example 2: Specific backend - Ollama
    print("\n2. Ollama backend:")
    try:
        vlm_ollama = VLMUtils(
            backend=VLMBackend.OLLAMA,
            ollama_url="http://localhost:11434",
            ollama_model="llava"
        )
        
        if vlm_ollama.is_available():
            result = vlm_ollama.process_image(
                "VisionLangAnnotateModels/sampledata/sjsupeople.jpg", 
                "Describe the people in this image."
            )
            print(f"Ollama result: {result}")
        else:
            print("Ollama backend not available")
            
    except Exception as e:
        print(f"Ollama error: {e}")
    
    # Example 3: Specific backend - vLLM Package
    print("\n3. vLLM Package backend:")
    try:
        vlm_package = VLMUtils(
            backend=VLMBackend.VLLM_PACKAGE,
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        if vlm_package.is_available():
            result = vlm_package.process_image(
                "VisionLangAnnotateModels/sampledata/sjsupeople.jpg", 
                "What activities are happening in this image?"
            )
            print(f"vLLM Package result: {result}")
        else:
            print("vLLM Package backend not available")
            
    except Exception as e:
        print(f"vLLM Package error: {e}")
    
    # Example 4: Multiple images
    print("\n4. Multiple images processing:")
    try:
        vlm = VLMUtils()
        
        if vlm.is_available():
            images = [
                "VisionLangAnnotateModels/sampledata/sjsupeople.jpg",
                "VisionLangAnnotateModels/sampledata/bus.jpg"
            ]
            prompts = [
                "Describe the people in this image.",
                "What type of vehicle is this?"
            ]
            
            results = vlm.process_multiple_images(images, prompts)
            
            # Get performance summary
            summary = vlm.get_performance_summary(results)
            print(f"Performance summary: {summary}")
            
        else:
            print("No backends available for multiple image processing")
            
    except Exception as e:
        print(f"Multiple images error: {e}")


if __name__ == "__main__":
    #example_usage()
    from vlm_utils import VLMUtils, VLMBackend

    # Use Ollama (currently working)
    vlm = VLMUtils(
        backend=VLMBackend.OLLAMA,
        ollama_model="qwen2.5vl:7b-q8_0"
    )

    # Auto-detect best available backend
    vlm = VLMUtils()  # Will try all backends and use the first available

    # Process images
    result = vlm.process_image("path/to/image.jpg", "Describe this image")