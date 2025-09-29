from abc import ABC, abstractmethod
import torch
from PIL import Image, ImageDraw, ImageFont
import time
import json
import os
import io
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

#install Flash attention: https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
#pip install flash-attn --no-build-isolation

#Qwen utilities for better handling of visual inputs:
#pip install qwen-vl-utils[decord]==0.0.8

# Try to import openai, but don't fail if it's not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not available. OpenAIVLM backend will not work.")

# Base imports for transformers
from transformers import Blip2Processor, Blip2ForConditionalGeneration
# Additional imports will be loaded dynamically based on model type

class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def generate(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Generate descriptions for images with given prompts."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the backend model."""
        pass

class HuggingFaceVLM(VLMBackend):
    """Hugging Face-based VLM backend with support for multiple model types.
    
    Supported model types:
    - BLIP2: Salesforce/blip2-opt-2.7b, Salesforce/blip2-flan-t5-xxl, etc.
    - LLaVA: llava-hf/llava-1.5-7b-hf, llava-hf/llava-v1.6-mistral-7b-hf, etc.
    - SmolVLM: HuggingFaceTB/SmolVLM-Instruct, HuggingFaceTB/SmolVLM-Base, etc.
    - GLM-4.5V: zai-org/GLM-4.5V (multimodal reasoning with thinking mode)
    - GLM-4.1V: zai-org/GLM-4.1V-9B-Thinking (multimodal reasoning model)
    - MiniGPT-4: Vision-CAIR/MiniGPT-4 (requires special handling)
    - GLIP: GLIPModel/GLIP (for object detection and grounding)
    """
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Determine model type based on model_name
        if "blip2" in model_name.lower():
            self.model_type = "blip2"
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype
            ).to(device)
        elif "llava" in model_name.lower():
            self.model_type = "llava"
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            # Special handling for LLaVA 1.6 Mistral to avoid Flash Attention 2.0 initialization issue
            if "llava-v1.6-mistral" in model_name.lower():
                self.processor = LlavaNextProcessor.from_pretrained(model_name)
                # # First initialize on CPU, then move to GPU to avoid Flash Attention 2.0 issues
                # self.model = LlavaNextForConditionalGeneration.from_pretrained(
                #     model_name,
                #     torch_dtype=self.dtype,
                #     low_cpu_mem_usage=True,
                #     use_flash_attention_2=True, #if device == "cuda"
                #     device_map=None  # Initialize on CPU first
                # )
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype
                )
                self.model = self.model.to(device)  # Then move to GPU
            else:
                self.processor = AutoProcessor.from_pretrained(model_name)
                # For other LLaVA models, use standard initialization
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype
                ).to(device)
        elif "qwen" in model_name.lower():
            self.model_type = "qwen"
            # Import Qwen-specific modules
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                self.processor = AutoProcessor.from_pretrained(model_name)
                
                # Initialize Qwen model with flash attention if available
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    attn_implementation="flash_attention_2" if device == "cuda" else "eager"
                ).to(device)
            except ImportError:
                raise ImportError("Please install the latest transformers version for Qwen2.5-VL support: "
                                 "pip install git+https://github.com/huggingface/transformers accelerate")
            except Exception as e:
                raise Exception(f"Error loading Qwen model: {str(e)}. Make sure you have the latest transformers version.")
        elif "smolvlm" in model_name.lower():
            self.model_type = "smolvlm"
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
                device_map=None
            ).to(device)
            
            # Explicitly move all model components to the target device
            if hasattr(self.model, 'vision_model'):
                self.model.vision_model = self.model.vision_model.to(device)
            if hasattr(self.model, 'language_model'):
                self.model.language_model = self.model.language_model.to(device)
            if hasattr(self.model, 'multi_modal_projector'):
                self.model.multi_modal_projector = self.model.multi_modal_projector.to(device)
        elif "minigpt-4" in model_name.lower():
            self.model_type = "minigpt4"
            # MiniGPT-4 uses BLIP-2's visual encoder with Vicuna LLM
            # This is a simplified implementation - full implementation would require the MiniGPT-4 repo
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=self.dtype
            ).to(device)
        elif "glm" in model_name.lower() and ("4.5v" in model_name.lower() or "4.1v" in model_name.lower()):
            if "4.5v" in model_name.lower():
                self.model_type = "glm4.5v"
                model_class_name = "Glm4vMoeForConditionalGeneration"
            else:  # GLM-4.1V
                self.model_type = "glm4.1v"
                model_class_name = "Glm4vForConditionalGeneration"
            
            # GLM-4.5V/4.1V multimodal reasoning models
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(model_name)
                
                # Import the appropriate model class
                if model_class_name == "Glm4vMoeForConditionalGeneration":
                    from transformers import Glm4vMoeForConditionalGeneration
                    model_class = Glm4vMoeForConditionalGeneration
                else:
                    from transformers import Glm4vForConditionalGeneration
                    model_class = Glm4vForConditionalGeneration
                
                # Initialize GLM model with flash attention if available
                self.model = model_class.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    attn_implementation="flash_attention_2" if device == "cuda" else "eager"
                ).to(device)
            except ImportError:
                raise ImportError(f"Please install the latest transformers version for {self.model_type.upper()} support: "
                                 "pip install git+https://github.com/huggingface/transformers.git")
            except Exception as e:
                raise Exception(f"Error loading {self.model_type.upper()} model: {str(e)}. Make sure you have the correct transformers version.")
        elif "glip" in model_name.lower():
            self.model_type = "glip"
            # GLIP is primarily for object detection and grounding
            # This is a simplified implementation - full implementation would require additional code
            from transformers import AutoProcessor, AutoModelForObjectDetection
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_name,
                torch_dtype=self.dtype
            ).to(device)
        else:
            # Default to BLIP2 for unknown models
            self.model_type = "blip2"
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype
            ).to(device)
    
    def generate(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        if self.model_type == "blip2":
            return self._generate_blip2(images, prompts)
        elif self.model_type == "llava":
            return self._generate_llava(images, prompts)
        elif self.model_type == "qwen":
            return self._generate_qwen(images, prompts)
        elif self.model_type == "smolvlm":
            return self._generate_smolvlm(images, prompts)
        elif self.model_type == "minigpt4":
            return self._generate_minigpt4(images, prompts)
        elif self.model_type in ["glm4.5v", "glm4.1v"]:
            return self._generate_glm4_5v(images, prompts)
        elif self.model_type == "glip":
            return self._generate_glip(images, prompts)
        else:
            # Default to BLIP2 generation
            return self._generate_blip2(images, prompts)
    
    def _generate_blip2(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Generate text using BLIP2 model."""
        results = []
        
        # Process each image-prompt pair individually
        for image, prompt in zip(images, prompts):
            # Prepare inputs using the processor
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            
            # Convert input tensors to the same dtype as the model
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(device=self.device, dtype=self.dtype)
                else:
                    inputs[k] = v.to(self.device)
            
            # Generate output with more generation parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # Increased from 100 for longer responses
                    min_length=5,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=2,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode result
            result = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "\n" in result:
                # Remove the first line and any newlines in the response
                result = result.split("\n", 1)[-1].strip()
                # Remove any remaining newlines
                result = result.replace("\n", " ")
            results.append(result)
        
        return results
    
    def _generate_llava(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Generate text using LLaVA model."""
        results = []
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            # Create conversation format for LLaVA
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template and prepare inputs
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Ensure all input tensors are on the correct device
            #This ensures that each tensor in the inputs dictionary is individually moved to the correct device, preventing the "Expected all tensors to be on the same device" error
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=300, pad_token_id=self.processor.tokenizer.eos_token_id)
            
            # Decode and append result
            try:
                # Try the standard approach first
                result = self.processor.batch_decode(output, skip_special_tokens=True)[0]
                #print(processor.decode(output[0], skip_special_tokens=True))
            except (ValueError, IndexError):
                # Handle the case where batch_decode returns a tuple with more than expected values
                # This is likely happening with LLaVA 1.6 Mistral
                if isinstance(output, torch.Tensor):
                    result = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    # If output is not a tensor, try to decode it directly
                    result = str(output)
            
            # Extract only the assistant's response
            if "ASSISTANT:" in result:
                result = result.split("ASSISTANT:")[-1].strip()
            # Handle Qwen [INST] format
            elif "[/INST]" in result: #used this
                result = result.split("[/INST]")[-1].strip()
            results.append(result)
        
        return results
    
    def _generate_smolvlm(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Generate text using SmolVLM model."""
        results = []
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            # Create conversation format for SmolVLM
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template and prepare inputs
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Ensure all input tensors are on the correct device with proper dtype
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                        inputs[k] = v.to(device=self.device, dtype=self.dtype)
                    else:
                        inputs[k] = v.to(device=self.device)
                        
            # Ensure the model is on the correct device
            self.model = self.model.to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=300)
            
            # Decode and append result
            try:
                # Try the standard approach first
                result = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            except (ValueError, IndexError):
                # Handle the case where batch_decode returns a tuple with more than expected values
                if isinstance(output, torch.Tensor):
                    result = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    # If output is not a tensor, try to decode it directly
                    result = str(output)
            
            # Extract only the assistant's response
            if "Assistant:" in result:
                result = result.split("Assistant:")[-1].strip()
            results.append(result)
        
        return results
    
    def _generate_minigpt4(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Generate text using MiniGPT-4 model."""
        # MiniGPT-4 implementation is similar to SmolVLM
        results = []
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            # Create conversation format for MiniGPT-4
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template and prepare inputs
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=300)
            
            # Decode and append result
            try:
                # Try the standard approach first
                result = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            except (ValueError, IndexError):
                # Handle the case where batch_decode returns a tuple with more than expected values
                if isinstance(output, torch.Tensor):
                    result = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    # If output is not a tensor, try to decode it directly
                    result = str(output)
            
            # Extract only the assistant's response
            if "ASSISTANT:" in result:
                result = result.split("ASSISTANT:")[-1].strip()
            results.append(result)
        
        return results
    
    def _generate_qwen(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Generate text using Qwen2.5-VL model.
        
        Optimized to process multiple images in a single batch when they are from the same source image.
        This optimization is particularly useful when processing multiple cropped regions from a single
        image, as it allows the model to see all regions at once and generate responses more efficiently.
        
        The method automatically detects when batch processing is appropriate and falls back to
        individual processing when necessary. For best results, ensure that:
        1. All images in the batch are from the same source image (e.g., cropped regions)
        2. The number of images matches the number of prompts
        3. Each prompt is specific to its corresponding image region
        
        The optimization can significantly reduce processing time when handling multiple regions
        from the same image, as it requires only one model inference instead of multiple separate calls.
        """
        # Check if we're processing multiple regions from the same image
        if len(images) <= 1 or len(images) != len(prompts):
            # Fall back to original implementation for single image or mismatched lists
            return self._generate_qwen_single(images, prompts)
        
        # Process multiple images in a single batch
        try:
            # Create conversation format for Qwen with multiple images
            conversation = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Add all images to the content, resizing small images if needed
            resized_images = []
            for image in images:
                # Check if image dimensions are smaller than the required factor of 28
                width, height = image.size
                if width < 28 or height < 28:
                    print(f"Resizing small image in _generate_qwen batch processing: {width}x{height} -> minimum factor of 28")
                    # Calculate new dimensions that are multiples of 28
                    new_width = max(28, ((width + 27) // 28) * 28)
                    new_height = max(28, ((height + 27) // 28) * 28)
                    print(f"New dimensions: {new_width}x{new_height}")
                    # Resize the image using a high-quality resampling method
                    try:
                        # For newer PIL versions
                        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    except AttributeError:
                        # For older PIL versions
                        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                    resized_images.append(resized_image)
                    conversation[0]["content"].append({"type": "image"})
                else:
                    resized_images.append(image)
                    conversation[0]["content"].append({"type": "image"})
            
            # Add the text prompt (using the first prompt as the main question)
            # We'll include all individual prompts in the text to maintain context
            combined_prompt = "For each region in the provided images, answer the following:\n"
            for i, prompt in enumerate(prompts):
                combined_prompt += f"Region {i+1}: {prompt}\n"
            
            conversation[0]["content"].append({"type": "text", "text": combined_prompt})
            
            # Apply chat template
            text_prompt = self.processor.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs with multiple images (using resized images)
            inputs = self.processor(
                text=[text_prompt],
                images=resized_images,  # Pass resized images as a list
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, 
                    max_new_tokens=512,
                    temperature=0.1,          # ✅ lower temp = more deterministic (good for detection)
                    top_p=0.8,                # optional
                    do_sample=False,          # ✅ deterministic output helps structured tasks
                    eos_token_id=self.processor.tokenizer.eos_token_id
                    )  # Increased token limit for multiple responses
            
            # Decode the output
            try:
                # Try to get just the new tokens (excluding input tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            except (ValueError, IndexError, AttributeError):
                # Fall back to standard decoding if trimming fails
                result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract only the assistant's response
            if "assistant:" in result.lower():
                result = re.split(r'assistant:', result, flags=re.IGNORECASE)[-1].strip()
            # Handle Qwen [INST] format
            elif "[/INST]" in result:
                result = result.split("[/INST]")[-1].strip()
            
            # Parse the result to extract individual responses for each region
            # This assumes the model follows our format of "Region X: [response]"
            results = []
            region_pattern = r'Region \d+:?\s*(.*?)(?=Region \d+:|$)'
            region_matches = re.finditer(region_pattern, result, re.DOTALL)
            
            for match in region_matches:
                results.append(match.group(1).strip())
            
            # If we couldn't parse individual responses, split the result evenly
            if not results:
                # Fall back to individual processing
                return self._generate_qwen_single(images, prompts)
            
            # Ensure we have a result for each prompt
            while len(results) < len(prompts):
                results.append("No response generated for this region.")
            
            # Trim to match the number of prompts
            results = results[:len(prompts)]
            
            return results
        
        except Exception as e:
            print(f"Error in batch Qwen processing: {str(e)}. Falling back to individual processing.")
            # Fall back to individual processing if batch processing fails
            return self._generate_qwen_single(images, prompts)
    
    def _generate_qwen_single(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Process images individually with Qwen model (original implementation)."""
        results = []
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            # Create conversation format for Qwen
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template and prepare inputs
            text_prompt = self.processor.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Check if image dimensions are smaller than the required factor of 28
            width, height = image.size
            if width < 28 or height < 28:
                print(f"Resizing small image in _generate_qwen_single: {width}x{height} -> minimum factor of 28")
                # Calculate new dimensions that are multiples of 28
                new_width = max(28, ((width + 27) // 28) * 28)
                new_height = max(28, ((height + 27) // 28) * 28)
                print(f"New dimensions: {new_width}x{new_height}")
                # Resize the image using a high-quality resampling method
                try:
                    # For newer PIL versions
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    # For older PIL versions
                    image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Process inputs
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=1000)
            
            # Decode and append result
            try:
                # Try the standard approach first
                result = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            except (ValueError, IndexError):
                # Handle the case where batch_decode returns a tuple with more than expected values
                if isinstance(output, torch.Tensor):
                    result = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    # If output is not a tensor, try to decode it directly
                    result = str(output)
            
            # Extract only the assistant's response
            if "assistant\n" in result.lower():
                result = re.split(r'assistant\n', result, flags=re.IGNORECASE)[-1].strip()
            results.append(result)
        
        return results
    
    def _generate_glip(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Generate text using GLIP model for object detection and grounding."""
        results = []
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            # GLIP is primarily for object detection and grounding
            # Here we're using it to detect objects mentioned in the prompt
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process detection results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results_processed = self.processor.post_process_object_detection(
                outputs, 
                threshold=0.5, 
                target_sizes=target_sizes
            )[0]
            
            # Format results as text
            detections = []
            for score, label, box in zip(results_processed["scores"], results_processed["labels"], results_processed["boxes"]):
                detections.append(f"{self.processor.tokenizer.decode(label)}: {score:.2f} at {box.tolist()}")
            
            results.append("\n".join(detections) if detections else "No objects detected.")
        
        return results
    
    def _generate_glm4_5v(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Generate text using GLM-4.5V model.
        
        GLM-4.5V is a multimodal reasoning model that supports thinking mode for enhanced reasoning.
        It can handle diverse visual content including images, videos, documents, and GUI tasks.
        """
        results = []
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            # Create conversation format for GLM-4.5V
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template and prepare inputs
            text_prompt = self.processor.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt"
            )
            
            # Ensure all input tensors are on the correct device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate response with GLM-4.5V specific parameters (increased tokens for longer responses)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,  # Increased from 100 to 500 for longer responses
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=2,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the output
            try:
                # Try to get just the new tokens (excluding input tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            except (ValueError, IndexError, AttributeError):
                # Fall back to standard decoding if trimming fails
                result = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract only the assistant's response
            # GLM-4.5V may use different response markers
            if "assistant" in result.lower():
                result = re.split(r'assistant:?', result, flags=re.IGNORECASE)[-1].strip()
            elif "[/INST]" in result:
                result = result.split("[/INST]")[-1].strip()
            elif "<|im_start|>assistant" in result:
                result = result.split("<|im_start|>assistant")[-1].strip()
                if result.startswith("\n"):
                    result = result[1:]
            
            # Clean up any remaining formatting
            result = result.strip()
            if result.endswith("<|im_end|>"):
                result = result[:-10].strip()
            
            # For GLM-4.1V-Thinking model, extract thinking and final response separately
            if "4.1v" in self.model_name.lower() and "thinking" in self.model_name.lower():
                thinking_content = ""
                final_response = result
                
                # Extract thinking content between <think> tags (handle incomplete tags)
                if '<think>' in result:
                    # Try to find complete <think>...</think> pattern first
                    think_pattern = r'<think>(.*?)</think>'
                    think_matches = re.findall(think_pattern, result, re.DOTALL)
                    
                    if think_matches:
                        # Complete thinking tags found
                        thinking_content = think_matches[0].strip()
                        final_response = re.sub(think_pattern, '', result, flags=re.DOTALL).strip()
                    else:
                        # Incomplete thinking tag - extract everything after <think>
                        think_start = result.find('<think>')
                        if think_start != -1:
                            thinking_content = result[think_start + 7:].strip()  # +7 for len('<think>')
                            final_response = result[:think_start].strip()
                
                # Store both thinking and final response
                if thinking_content:
                    result = f"**Thinking Process:**\n{thinking_content}\n\n**Final Response:**\n{final_response}"
            
            results.append(result)
        
        return results
    
    def parse_object_detection_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse object detection response text into structured JSON format.
        Extracts objects, descriptions, and bounding box coordinates.
        """
        objects = []
        
        # Pattern to match bounding box coordinates: (x1, y1, x2, y2)
        bbox_pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        
        # Split response into sections
        lines = response_text.split('\n')
        current_object = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for object headers (e.g., "Person 1", "Building on the Left")
            if line.startswith('- **') and line.endswith(':**'):
                # Extract object name
                object_name = line[4:-3]  # Remove "- **" and ":**"
                current_object = {
                    'name': object_name,
                    'description': '',
                    'bbox': None,
                    'confidence': 1.0  # Default confidence
                }
            
            # Look for bounding box coordinates
            elif '**Bounding Box:**' in line:
                bbox_match = re.search(bbox_pattern, line)
                if bbox_match and current_object:
                    x1, y1, x2, y2 = map(int, bbox_match.groups())
                    current_object['bbox'] = [x1, y1, x2, y2]
            
            # Look for descriptions
            elif '**Description:**' in line:
                description = line.split('**Description:**')[-1].strip()
                if current_object:
                    current_object['description'] = description
                    objects.append(current_object)
                    current_object = None
        
        # Alternative parsing for simpler formats
        if not objects:
            # Try to parse simpler format with direct coordinate mentions
            bbox_matches = re.findall(r'([^\n]+?)\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', response_text)
            for match in bbox_matches:
                description, x1, y1, x2, y2 = match
                objects.append({
                    'name': description.strip().split(':')[0] if ':' in description else description.strip(),
                    'description': description.strip(),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': 1.0
                })
        
        return {
            'objects': objects,
            'total_objects': len(objects),
            'image_info': {
                'format': 'bbox',
                'coordinate_system': 'absolute'
            }
        }
    
    def visualize_detections(self, image: Image.Image, detection_results: Dict[str, Any], 
                           save_path: Optional[str] = None) -> Image.Image:
        """
        Visualize object detection results on the image with bounding boxes and labels.
        """
        # Create a copy of the image to draw on
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Color palette for different objects
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000080'
        ]
        
        objects = detection_results.get('objects', [])
        
        for i, obj in enumerate(objects):
            if obj.get('bbox'):
                x1, y1, x2, y2 = obj['bbox']
                color = colors[i % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Prepare label text
                label = obj.get('name', f'Object {i+1}')
                confidence = obj.get('confidence', 1.0)
                label_text = f"{label} ({confidence:.2f})" if confidence < 1.0 else label
                
                # Calculate text size and position
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw label background
                label_bg = [x1, y1 - text_height - 4, x1 + text_width + 8, y1]
                draw.rectangle(label_bg, fill=color)
                
                # Draw label text
                draw.text((x1 + 4, y1 - text_height - 2), label_text, fill='white', font=font)
        
        # Save if path provided
        if save_path:
            vis_image.save(save_path)
        
        return vis_image
    
    def generate_with_structured_output(self, images: List[Image.Image], prompts: List[str], 
                                      parse_objects: bool = False, 
                                      visualize: bool = False,
                                      save_visualizations: bool = False,
                                      output_dir: str = "./detections") -> List[Dict[str, Any]]:
        """
        Generate responses and optionally parse object detection results and create visualizations.
        
        Args:
            images: List of PIL Images
            prompts: List of text prompts
            parse_objects: Whether to parse object detection results
            visualize: Whether to create visualization images
            save_visualizations: Whether to save visualization images
            output_dir: Directory to save visualizations
            
        Returns:
            List of dictionaries containing response text, parsed objects, and visualization info
        """
        # Generate text responses
        text_responses = self.generate(images, prompts)
        
        results = []
        
        # Create output directory if needed
        if save_visualizations:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, (image, prompt, response) in enumerate(zip(images, prompts, text_responses)):
            result = {
                'prompt': prompt,
                'response': response,
                'parsed_objects': None,
                'visualization_path': None,
                'visualization_image': None
            }
            
            # Parse objects if requested
            if parse_objects:
                try:
                    parsed = self.parse_object_detection_response(response)
                    result['parsed_objects'] = parsed
                except Exception as e:
                    print(f"Warning: Failed to parse objects for image {i}: {e}")
                    result['parsed_objects'] = {'objects': [], 'total_objects': 0}
            
            # Create visualization if requested
            if visualize and result['parsed_objects'] and result['parsed_objects']['objects']:
                try:
                    vis_path = None
                    if save_visualizations:
                        vis_path = os.path.join(output_dir, f"detection_result_{i}.png")
                    
                    vis_image = self.visualize_detections(image, result['parsed_objects'], vis_path)
                    result['visualization_image'] = vis_image
                    result['visualization_path'] = vis_path
                except Exception as e:
                    print(f"Warning: Failed to create visualization for image {i}: {e}")
            
            results.append(result)
        
        return results
    
    def get_name(self) -> str:
        return f"HuggingFace-{self.model_type}-{self.model_name.split('/')[-1]}"

class OpenAIVLM(VLMBackend):
    """OpenAI-based VLM backend (GPT-4V)."""
    
    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not available. Please install it with 'pip install openai'.")
        
        # Import is done at the module level with try/except
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        if not OPENAI_AVAILABLE:
            return ["Error: OpenAI package not available"] * len(images)
        
        results = []
        for image, prompt in zip(images, prompts):
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_byte_arr}"}}
                            ]
                        }
                    ],
                    max_tokens=50
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        return results
    
    def get_name(self) -> str:
        return "OpenAI-GPT4V"

class VLMClassifier:
    """Enhanced VLM Classifier with multiple backends and evaluation capabilities."""
    
    def __init__(self, backend: VLMBackend):
        self.backend = backend
    
    def embed_box_on_image(self, image: Image.Image, bbox: Tuple[int, int, int, int], 
                          color: str = "red", width: int = 4) -> Image.Image:
        """Draw bounding box on image for visual context."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=color, width=width)
        return img
    
    def two_view_image(self, crop: Image.Image, full_context: Image.Image) -> Image.Image:
        """Create a two-view image combining crop and context."""
        context_resized = full_context.resize(crop.size)
        combined = Image.new("RGB", (crop.width * 2, crop.height))
        combined.paste(crop, (0, 0))
        combined.paste(context_resized, (crop.width, 0))
        return combined
    
    def classify(self, image_crops_with_prompts: List[Tuple[Image.Image, str]]) -> List[str]:
        """Basic classification of image crops."""
        images, prompts = zip(*image_crops_with_prompts)
        return self.backend.generate(list(images), list(prompts))
    
    def classify_relativepos(self, object_crops: List[Image.Image], 
                           full_context_images: List[Image.Image], 
                           prompts: List[str]) -> List[str]:
        """Classification with relative position context."""
        context_prompts = [p + " The object is marked in the full image." for p in prompts]
        return self.backend.generate(full_context_images, context_prompts)
    
    def classify_overlay(self, full_context_images_with_bbox: List[Image.Image], 
                        prompts: List[str]) -> List[str]:
        """Classification with bounding box overlay."""
        overlay_prompts = [p + " The object is highlighted in the image with a red box." for p in prompts]
        return self.backend.generate(full_context_images_with_bbox, overlay_prompts)
    
    def classify_twoview(self, crops: List[Image.Image], 
                        full_contexts: List[Image.Image], 
                        prompts: List[str]) -> List[str]:
        """Classification with two-view context."""
        image_pairs = [self.two_view_image(crop, context) 
                      for crop, context in zip(crops, full_contexts)]
        return self.backend.generate(image_pairs, prompts)
    
    def classify_batch(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Batch classification of images."""
        return self.backend.generate(images, prompts)
    
    def benchmark(self, test_cases: List[Dict[str, Any]], 
                  methods: List[str] = None) -> Dict[str, Any]:
        """Benchmark different classification methods.
        
        Args:
            test_cases: List of test cases, each containing:
                - image: PIL Image
                - bbox: Optional[Tuple[int, int, int, int]]
                - prompt: str
            methods: List of methods to benchmark ("basic", "relativepos", "overlay", "twoview")
                    Note: If bbox is not provided, only "basic" method will be used
        
        Returns:
            Dictionary containing benchmark results.
        """
        if methods is None:
            methods = ["basic", "relativepos", "overlay", "twoview"]
        
        results = {
            "model": self.backend.get_name(),
            "methods": {},
            "timing": {}
        }
        
        for method in methods:
            method_results = []
            method_times = []
            
            for case in test_cases:
                image = case["image"]
                bbox = case.get("bbox")  # Make bbox optional
                prompt = case["prompt"]
                
                # Skip non-basic methods if no bbox is provided
                if bbox is None and method != "basic":
                    continue
                
                # Prepare inputs based on method
                if bbox is not None:
                    crop = image.crop(bbox)
                    bbox_image = self.embed_box_on_image(image, bbox)
                else:
                    crop = image  # Use whole image if no bbox
                    bbox_image = image
                
                start_time = time.time()
                
                result = None
                if method == "basic":
                    result = self.classify([(crop, prompt)])[0]
                elif bbox is not None:  # Only execute these methods if bbox exists
                    if method == "relativepos":
                        result = self.classify_relativepos([crop], [bbox_image], [prompt])[0]
                    elif method == "overlay":
                        result = self.classify_overlay([bbox_image], [prompt])[0]
                    elif method == "twoview":
                        result = self.classify_twoview([crop], [image], [prompt])[0]
                
                end_time = time.time()
                method_times.append(end_time - start_time)
                
                method_results.append({
                    "prompt": prompt,
                    "result": result,
                    "time": end_time - start_time,
                    "has_bbox": bbox is not None
                })
            
            # Only add method results if there are any results for this method
            if method_results:
                results["methods"][method] = method_results
                results["timing"][method] = {
                    "avg": sum(method_times) / len(method_times),
                    "min": min(method_times),
                    "max": max(method_times)
                }
        
        return results

def test_vlm_comparison(image_path: str, bbox: Tuple[int, int, int, int], 
                       prompt: str, save_dir: Optional[str] = None) -> None:
    """Test and compare different VLM backends and classification methods.
    
    Args:
        image_path: Path to the test image
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        prompt: Base prompt for classification
        save_dir: Optional directory to save results
    """
    # Load test image
    image = Image.open(image_path)
    
    # Prepare test case
    test_case = {
        "image": image,
        "bbox": bbox,
        "prompt": prompt
    }
    
    # Initialize backends
    backends = [
        HuggingFaceVLM("Salesforce/blip2-opt-2.7b"),
        #HuggingFaceVLM("Salesforce/blip2-flan-t5-xl"),
        # Uncomment to test with OpenAI (requires API key)
        # OpenAIVLM(os.getenv("OPENAI_API_KEY"))
    ]
    
    # Methods to test
    methods = ["basic", "relativepos", "overlay", "twoview"]
    
    # Run benchmarks
    all_results = {}
    for backend in backends:
        classifier = VLMClassifier(backend)
        results = classifier.benchmark([test_case], methods)
        all_results[backend.get_name()] = results
    
    # Save or print results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vlm_comparison_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\nVLM Comparison Results:")
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        for method, timing in results["timing"].items():
            print(f"  {method}:")
            print(f"    Average time: {timing['avg']:.3f}s")
            print(f"    Result: {results['methods'][method][0]['result']}")

def testBlip2():
    # pip install accelerate
    import torch
    import requests
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    # Load BLIP-2 processor and model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Load your image
    img_path = "output/gcs_sources/Sweeper 19303/20250224/064224_000100.jpg"
    image = Image.open(img_path).convert("RGB")

    # Define your prompt (keep it concise and grounded in visual context)
    prompt = "Does the image show dumped trash, glass, yard waste, or debris? If there is a trash container. Determine whether it is a residential bin or a commercial dumpster, and whether it is improperly placed—especially if it obstructs a bike lane or pedestrian path."

    # Prepare inputs using the processor
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    for k, v in inputs.items():
        if v.dtype == torch.float:
            inputs[k] = v.to(model.device, dtype=torch.float16)
        else:
            inputs[k] = v.to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            min_length=5,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    # Decode result
    output = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("🧠 Model Output:", output)

def testBlip2_module():
    import torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    # ---------- Setup ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    # ---------- Image Load ----------
    img_path = "output/gcs_sources/Sweeper 19303/20250224/064224_000100.jpg"
    image = Image.open(img_path).convert("RGB")

    # ---------- Helper Function ----------
    def generate_response(image, prompt, max_new_tokens=80):
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        for k, v in inputs.items():
            if v.dtype == torch.float: #'pixel_values'
                inputs[k] = v.to(model.device, dtype=torch.float16)
            else:#torch.int64
                inputs[k] = v.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                no_repeat_ngram_size=2,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        return processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ---------- Modular Prompts ----------
    prompt_trash = (
        "Does this image show dumped trash, broken glass, yard waste, or debris? "
        "List all that apply or say 'none'."
    )
    prompt_container = (
        "Is there a trash container in the image? If so, is it a residential bin or a commercial dumpster?"
    )
    prompt_placement = (
        "Is the trash container placed improperly, such as blocking a bike lane or pedestrian path?"
    )

    # ---------- Run Model ----------
    result = {
        "trash_type": generate_response(image, prompt_trash),
        "container_type": generate_response(image, prompt_container),
        "improper_placement": generate_response(image, prompt_placement)
    }

    # ---------- Output ----------
    print("🧾 Parsed Result:")
    for k, v in result.items():
        print(f"{k}: {v}")


def test_multi_model_vlm(img_path):
    """Test the enhanced HuggingFaceVLM backend with different model types."""
    import os
    import gc
    from PIL import Image
    
    # Sample image path - adjust as needed
    #img_path = "output/gcs_sources/Sweeper 19303/20250224/064224_000100.jpg"
    if not os.path.exists(img_path):
        print(f"Warning: Image path {img_path} not found. Using a sample image.")
        # Create a simple test image if the specified path doesn't exist
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 400, 400], outline='black', width=5)
        draw.text((200, 250), "Test Image", fill='black')
    else:
        img = Image.open(img_path).convert("RGB")
    
    # Test prompt
    #prompt = "Describe what you see in this image."
    prompt = 'Please analyze this image in great detail. Describe all the objects you can see and describe their locations in bounding box coordinate. '
    
    # Model configurations to test
    # Note: Uncomment models as needed, but be aware of memory requirements
    model_configs = [
        # Default BLIP2 model
        {"name": "Salesforce/blip2-opt-2.7b", "description": "BLIP2 (Default)"},
        
        # LLaVA models
        # {"name": "llava-hf/llava-1.5-7b-hf", "description": "LLaVA 1.5 (7B)"},
        {"name": "llava-hf/llava-v1.6-mistral-7b-hf", "description": "LLaVA 1.6 Mistral (7B)"},
        
        # SmolVLM models - temporarily disabled due to device mismatch issues
        {"name": "HuggingFaceTB/SmolVLM-Instruct", "description": "SmolVLM Instruct"},
        
        # Qwen models
        {"name": "Qwen/Qwen2.5-VL-7B-Instruct", "description": "Qwen2.5-VL (7B)"},
        
        # GLM-4.5V model
        #{"name": "zai-org/GLM-4.5V", "description": "GLM-4.5V"},
        
        # GLM-4.1V model
        {"name": "zai-org/GLM-4.1V-9B-Thinking", "description": "GLM-4.1V"},
        
        # MiniGPT-4 and GLIP would require additional setup
        # {"name": "Vision-CAIR/MiniGPT-4", "description": "MiniGPT-4"},
        #{"name": "GLIPModel/GLIP", "description": "GLIP"}
    ]
    
    # Test each model configuration
    results = {}
    for config in model_configs:
        print(f"\nTesting {config['description']} ({config['name']})...")
        vlm = None
        try:
            # Initialize the model
            vlm = HuggingFaceVLM(model_name=config['name'])
            
            # Generate response
            start_time = time.time()
            response = vlm.generate([img], [prompt])[0]
            end_time = time.time()
            
            # Store results
            results[config['description']] = {
                "response": response,
                "time": end_time - start_time
            }
            
            # Print results
            print(f"Response: {response}")
            print(f"Time: {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error with {config['description']}: {str(e)}")
            results[config['description']] = {"error": str(e)}
        
        finally:
            # Explicitly delete the model object and clean up GPU memory
            if vlm is not None:
                if hasattr(vlm, 'model') and vlm.model is not None:
                    del vlm.model
                if hasattr(vlm, 'processor') and vlm.processor is not None:
                    del vlm.processor
                del vlm
            
            # Aggressive GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            print("GPU memory cleaned up.")
    
    return results

# Example usage
if __name__ == "__main__":
    # Uncomment the test function you want to run
    
    # Test the enhanced multi-model VLM backend
    #test_multi_model_vlm(img_path="VisionLangAnnotateModels/sampledata/vehiclecrop.jpeg")
    test_multi_model_vlm(img_path="VisionLangAnnotateModels/sampledata/sjsupeople.jpg")
    
    # Test the original BLIP2 module
    # testBlip2_module()
    
    # Test VLM comparison
    # test_vlm_comparison(
    #     image_path="output/gcs_sources/Sweeper 19303/20250224/064224_000100.jpg",
    #     bbox=None,
    #     prompt="Analyze the vehicle in the image. Is it abandoned, burned, blocking a bike lane, fire hydrant, or red curb? Is it missing parts like tires or windows, or up on jacks?",
    #     save_dir="./results"
    # )