import json
import re
import requests
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from PIL import Image
import time


def extract_ollama_response_text(response_data: Dict[str, Any]) -> str:
    """
    Extract the response text from an Ollama API response, handling both generate and chat API formats.
    
    Args:
        response_data: The JSON response from the Ollama API
        
    Returns:
        The extracted response text
    """
    # Check if this is a chat API response (which has a different structure)
    if "message" in response_data and "content" in response_data["message"]:
        # Chat API format: response is in message.content
        response_text = response_data["message"]["content"]
    else:
        # Generate API format: response is in the response field
        response_text = response_data.get("response", "")
        
    # Extract only the assistant's response if needed
    if "assistant:" in response_text.lower():
        response_text = re.split(r'assistant:', response_text, flags=re.IGNORECASE)[-1].strip()
    # Handle Qwen [INST] format
    elif "[/INST]" in response_text:
        response_text = response_text.split("[/INST]")[-1].strip()
        
    return response_text

# Default Ollama API endpoint (for older versions)
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Default endpoint for chat-based models (preferred for Ollama 0.9.6+)
DEFAULT_OLLAMA_CHAT_API_URL = "http://localhost:11434/api/chat"

class OllamaClient:
    """
    Client for interacting with Ollama API for both text-only and vision LLMs.
    
    This class provides interfaces for:
    1. Text-only LLMs (process_text)
    2. Vision LLMs (process_vision)
    
    It handles API communication, error handling, and response parsing.
    """
    
    def __init__(self, api_url: str = DEFAULT_OLLAMA_API_URL, chat_api_url: str = DEFAULT_OLLAMA_CHAT_API_URL, timeout: int = 60, stream_callback: Optional[callable] = None):
        """
        Initialize the Ollama client.
        
        Args:
            api_url: URL of the Ollama API endpoint for generate requests
            chat_api_url: URL of the Ollama API endpoint for chat requests
            timeout: Timeout in seconds for API requests
            stream_callback: Optional callback function for streaming responses. 
                             The callback should accept a string chunk and return None.
        """
        self.api_url = api_url
        self.chat_api_url = chat_api_url
        self.timeout = timeout
        self.stream_callback = stream_callback
    
    def process_text(self, 
                     prompt: str, 
                     model: str, 
                     format_json: bool = True,
                     system_prompt: Optional[str] = None,
                     stream: bool = False) -> Dict[str, Any]:
        """
        Process text using a text-only Ollama LLM.
        
        Args:
            prompt: The text prompt to send to the model
            model: Name of the Ollama model to use (e.g., "llama3")
            format_json: Whether to request JSON format output
            system_prompt: Optional system prompt to set context
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        # Check if streaming is requested but no callback is set
        if stream and self.stream_callback is None:
            return {
                "success": False,
                "error": "Stream callback must be set to use streaming"
            }
            
        try:
            # Prepare the request payload for the chat API
            chat_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": stream
            }
            
            # Add optional parameters
            if format_json:
                chat_payload["format"] = "json"
                
            if system_prompt:
                chat_payload["system"] = system_prompt
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(chat_payload, is_chat=True, fallback_to_generate=True)
            
            # For non-streaming responses
            # Call Ollama Chat API first (preferred for newer models)
            try:
                response = requests.post(
                    self.chat_api_url, 
                    json=chat_payload,
                    timeout=self.timeout
                )
                
                # If chat API fails, fall back to generate API
                if response.status_code != 200:
                    # Prepare the request payload for the generate API
                    generate_payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                    }
                    
                    # Add optional parameters
                    if format_json:
                        generate_payload["format"] = "json"
                        
                    if system_prompt:
                        generate_payload["system"] = system_prompt
                    
                    # Call Ollama Generate API
                    response = requests.post(
                        self.api_url, 
                        json=generate_payload,
                        timeout=self.timeout
                    )
            except Exception as e:
                # If chat API fails with an exception, try the generate API
                generate_payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                }
                
                # Add optional parameters
                if format_json:
                    generate_payload["format"] = "json"
                    
                if system_prompt:
                    generate_payload["system"] = system_prompt
                
                # Call Ollama Generate API
                response = requests.post(
                    self.api_url, 
                    json=generate_payload,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                
                # Extract response text using the utility function
                response_text = extract_ollama_response_text(response_data)
                
                print(f"DEBUG - Raw API response: {response_data}")
                print(f"DEBUG - Extracted response text: {response_text}")
                
                return {
                    "success": True,
                    "response": response_text,
                    "raw_response": response_data
                }
            else:
                # Handle API error
                error_msg = f"Ollama API error: {response.status_code}"
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except requests.exceptions.Timeout:
            # Handle timeout
            error_msg = f"Timeout while calling Ollama API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error calling Ollama: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _handle_streaming_request(self, payload: Dict[str, Any], is_chat: bool = True, fallback_to_generate: bool = True) -> Dict[str, Any]:
        """
        Handle streaming requests to Ollama API.
        
        Args:
            payload: The request payload
            is_chat: Whether to use the chat API (True) or generate API (False)
            fallback_to_generate: Whether to fall back to generate API if chat API fails
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Ensure stream is set to True in the payload
            payload["stream"] = True
            
            # Determine which API to use
            api_url = self.chat_api_url if is_chat else self.api_url
            
            # Make the streaming request
            with requests.post(api_url, json=payload, stream=True, timeout=self.timeout) as response:
                if response.status_code != 200:
                    # If chat API fails and fallback is enabled, try generate API
                    if is_chat and fallback_to_generate:
                        # Convert chat payload to generate payload if needed
                        if "messages" in payload:
                            # Extract prompt from messages
                            messages = payload.get("messages", [])
                            if messages and "content" in messages[0]:
                                generate_payload = payload.copy()
                                generate_payload["prompt"] = messages[0]["content"]
                                del generate_payload["messages"]
                                
                                # Try again with generate API
                                return self._handle_streaming_request(generate_payload, is_chat=False, fallback_to_generate=False)
                    
                    # Handle API error
                    error_msg = f"Ollama API error: {response.status_code}"
                    return {
                        "success": False,
                        "error": error_msg
                    }
                
                # Process the streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        # Parse the JSON line
                        try:
                            chunk = json.loads(line)
                            
                            # Extract the text chunk
                            if is_chat and "message" in chunk and "content" in chunk["message"]:
                                # Chat API format
                                text_chunk = chunk["message"]["content"]
                            else:
                                # Generate API format
                                text_chunk = chunk.get("response", "")
                            
                            # Call the callback with the chunk
                            if self.stream_callback and text_chunk:
                                self.stream_callback(text_chunk)
                            
                            # Append to full response
                            full_response += text_chunk
                            
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
                
                # Return the full response
                return {
                    "success": True,
                    "response": full_response,
                    "raw_response": {"response": full_response}  # Simplified raw response
                }
                
        except requests.exceptions.Timeout:
            # Handle timeout
            error_msg = f"Timeout while streaming from Ollama API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error streaming from Ollama: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def process_vision(self,
                      images: List[Image.Image],
                      prompts: List[str],
                      model: str,
                      combined_prompt: Optional[str] = None,
                      stream: bool = False) -> Dict[str, Any]:
        """
        Process images using a vision-capable Ollama LLM (e.g., Qwen2.5-VL).
        
        This method supports both single and multiple images with corresponding prompts.
        For multiple images, it can either process them individually or as a batch with a
        combined prompt.
        
        Args:
            images: List of PIL Image objects
            prompts: List of text prompts corresponding to each image
            model: Name of the Ollama vision model to use (e.g., "qwen2.5-vl")
            combined_prompt: Optional combined prompt for batch processing
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed results or error information
        """
        # Validate inputs
        if not images or len(images) != len(prompts):
            return {
                "success": False,
                "error": "Number of images must match number of prompts"
            }
        
        # Check if streaming is requested but no callback is set
        if stream and self.stream_callback is None:
            return {
                "success": False,
                "error": "Stream callback must be set to use streaming"
            }
        
        try:
            # For single image processing
            if len(images) == 1:
                return self._process_single_vision(images[0], prompts[0], model, stream)
            
            # For multiple images
            if combined_prompt:
                # Process as batch with combined prompt
                return self._process_batch_vision(images, combined_prompt, model, stream)
            else:
                # Process each image individually and combine results
                results = []
                for i, (image, prompt) in enumerate(zip(images, prompts)):
                    result = self._process_single_vision(image, prompt, model, stream)
                    results.append(result)
                
                # Check if all were successful
                all_successful = all(r.get("success", False) for r in results)
                
                if all_successful:
                    return {
                        "success": True,
                        "responses": [r.get("response", "") for r in results],
                        "raw_responses": [r.get("raw_response", {}) for r in results]
                    }
                else:
                    # Return the first error encountered
                    for result in results:
                        if not result.get("success", False):
                            return result
                    
                    # Fallback error if none found but success is still False
                    return {
                        "success": False,
                        "error": "Unknown error in batch processing"
                    }
                
        except Exception as e:
            # Handle other errors
            error_msg = f"Error in vision processing: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _process_single_vision(self, image: Image.Image, prompt: str, model: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a single image with a vision-capable Ollama model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            model: Name of the Ollama vision model
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Resize image if needed (Qwen models require dimensions to be multiples of 28)
            width, height = image.size
            if width < 28 or height < 28 or width % 28 != 0 or height % 28 != 0:
                # Calculate new dimensions that are multiples of 28
                new_width = max(28, ((width + 27) // 28) * 28)
                new_height = max(28, ((height + 27) // 28) * 28)
                
                # Resize the image using a high-quality resampling method
                try:
                    # For newer PIL versions
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    # For older PIL versions
                    image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert image to base64
            import base64
            from io import BytesIO
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Prepare the request payload for the chat API
            chat_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_base64]
                    }
                ],
                "stream": stream
            }
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(chat_payload, is_chat=True, fallback_to_generate=True)
            
            # For non-streaming responses
            # Call Ollama Chat API first (preferred for vision models)
            try:
                response = requests.post(
                    self.chat_api_url, 
                    json=chat_payload,
                    timeout=self.timeout
                )
                
                # If chat API fails, fall back to generate API
                if response.status_code != 200:
                    # Prepare the request payload for the generate API
                    generate_payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "images": [img_base64]
                    }
                    
                    # Call Ollama Generate API
                    response = requests.post(
                        self.api_url, 
                        json=generate_payload,
                        timeout=self.timeout
                    )
            except Exception as e:
                # If chat API fails with an exception, try the generate API
                generate_payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "images": [img_base64]
                }
                
                # Call Ollama Generate API
                response = requests.post(
                    self.api_url, 
                    json=generate_payload,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                
                # Extract response text using the utility function
                response_text = extract_ollama_response_text(response_data)
                
                return {
                    "success": True,
                    "response": response_text,
                    "raw_response": response_data
                }
            else:
                # Handle API error
                error_msg = f"Ollama API error: {response.status_code}"
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except requests.exceptions.Timeout:
            # Handle timeout
            error_msg = f"Timeout while calling Ollama API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error calling Ollama vision: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _process_batch_vision(self, images: List[Image.Image], combined_prompt: str, model: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process multiple images as a batch with a combined prompt.
        
        Args:
            images: List of PIL Image objects
            combined_prompt: Combined prompt for all images
            model: Name of the Ollama vision model
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Convert all images to base64
            import base64
            from io import BytesIO
            
            img_base64_list = []
            for image in images:
                # Resize image if needed (Qwen models require dimensions to be multiples of 28)
                width, height = image.size
                if width < 28 or height < 28 or width % 28 != 0 or height % 28 != 0:
                    # Calculate new dimensions that are multiples of 28
                    new_width = max(28, ((width + 27) // 28) * 28)
                    new_height = max(28, ((height + 27) // 28) * 28)
                    
                    # Resize the image using a high-quality resampling method
                    try:
                        # For newer PIL versions
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    except AttributeError:
                        # For older PIL versions
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_base64_list.append(img_base64)
            
            # Prepare the request payload for the chat API
            chat_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": combined_prompt,
                        "images": img_base64_list
                    }
                ],
                "stream": stream
            }
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(chat_payload, is_chat=True, fallback_to_generate=True)
            
            # For non-streaming responses
            # Call Ollama Chat API first (preferred for vision models)
            try:
                response = requests.post(
                    self.chat_api_url, 
                    json=chat_payload,
                    timeout=self.timeout
                )
                
                # If chat API fails, fall back to generate API
                if response.status_code != 200:
                    # Prepare the request payload for the generate API
                    generate_payload = {
                        "model": model,
                        "prompt": combined_prompt,
                        "stream": False,
                        "images": img_base64_list
                    }
                    
                    # Call Ollama Generate API
                    response = requests.post(
                        self.api_url, 
                        json=generate_payload,
                        timeout=self.timeout
                    )
            except Exception as e:
                # If chat API fails with an exception, try the generate API
                generate_payload = {
                    "model": model,
                    "prompt": combined_prompt,
                    "stream": False,
                    "images": img_base64_list
                }
                
                # Call Ollama Generate API
                response = requests.post(
                    self.api_url, 
                    json=generate_payload,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                
                # Extract response text using the utility function
                response_text = extract_ollama_response_text(response_data)
                
                return {
                    "success": True,
                    "response": response_text,
                    "raw_response": response_data
                }
            else:
                # Handle API error
                error_msg = f"Ollama API error: {response.status_code}"
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except requests.exceptions.Timeout:
            # Handle timeout
            error_msg = f"Timeout while calling Ollama API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error calling Ollama batch vision: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }


def process_with_ollama_text(prompt: str, 
                           ollama_model: str, 
                           format_json: bool = True,
                           system_prompt: Optional[str] = None,
                           api_url: str = DEFAULT_OLLAMA_API_URL,
                           chat_api_url: str = DEFAULT_OLLAMA_CHAT_API_URL,
                           timeout: int = 60,
                           stream: bool = False,
                           stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Process text using a text-only Ollama LLM.
    
    Args:
        prompt: The text prompt to send to the model
        ollama_model: Name of the Ollama model to use (e.g., "llama3")
        format_json: Whether to request JSON format output
        system_prompt: Optional system prompt to set context
        api_url: URL of the Ollama API endpoint for generate requests
        chat_api_url: URL of the Ollama API endpoint for chat requests
        timeout: Timeout in seconds for API requests
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        Dictionary containing the processed result or error information
    """
    client = OllamaClient(api_url=api_url, chat_api_url=chat_api_url, timeout=timeout, stream_callback=stream_callback)
    return client.process_text(prompt, ollama_model, format_json, system_prompt, stream=stream)


def process_with_ollama_vision(images: List[Image.Image], 
                             prompts: List[str], 
                             ollama_model: str,
                             combined_prompt: Optional[str] = None,
                             api_url: str = DEFAULT_OLLAMA_API_URL,
                             chat_api_url: str = DEFAULT_OLLAMA_CHAT_API_URL,
                             timeout: int = 60,
                             stream: bool = False,
                             stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Process images using a vision-capable Ollama LLM.
    
    Args:
        images: List of PIL Image objects
        prompts: List of text prompts corresponding to each image
        ollama_model: Name of the Ollama vision model to use (e.g., "qwen2.5-vl")
        combined_prompt: Optional combined prompt for batch processing
        api_url: URL of the Ollama API endpoint for generate requests
        chat_api_url: URL of the Ollama API endpoint for chat requests
        timeout: Timeout in seconds for API requests
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        Dictionary containing the processed results or error information
    """
    client = OllamaClient(api_url=api_url, chat_api_url=chat_api_url, timeout=timeout, stream_callback=stream_callback)
    return client.process_vision(images, prompts, ollama_model, combined_prompt, stream=stream)


def process_with_ollama(descriptions: List[str], 
                       step1_labels: List[str], 
                       image_paths: List[str], 
                       ollama_model: str, 
                       allowed_classes: List[str] = None,
                       stream: bool = False,
                       stream_callback: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """
    Process VLM descriptions using a local Ollama model to standardize outputs.
    Processes all objects from a single image together to save processing time.
    
    Args:
        descriptions: List of VLM descriptions to process
        step1_labels: List of step1 detection labels corresponding to each description
        image_paths: List of image paths corresponding to each description
        ollama_model: Name of the Ollama model to use (e.g., "llama3")
        allowed_classes: List of allowed class names for standardization
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        List of dictionaries containing processed results
    """
    if allowed_classes is None:
        allowed_classes = [
            "Car", "Truck", "Vehicle blocking bike lane", "Burned vehicle", "Police car", 
            "Pedestrian", "Worker", "Street vendor", "Residential trash bin", "Commercial dumpster", 
            "Street sign", "Construction sign", "Traffic signal light", "Broken traffic lights", 
            "Tree", "Overhanging branch", "Dumped trash", "Yard waste", "Glass/debris", 
            "Pothole", "Unclear bike lane markings", "Utility pole", "Downed bollard", 
            "Cone", "Streetlight outage", "Graffiti", "Bench", "Vehicle in bike lane", 
            "Bicycle", "Scooter", "Wheelchair", "Bus", "Train", "Ambulance", "Fire truck", "Other"
        ]
    
    results = []
    
    # Group descriptions by image path
    image_groups = {}
    for i, (description, step1_label, image_path) in enumerate(zip(descriptions, step1_labels, image_paths)):
        if image_path not in image_groups:
            image_groups[image_path] = []
        image_groups[image_path].append({
            "index": i,
            "description": description,
            "step1_label": step1_label
        })
    
    # Process each image's objects together
    for image_path, objects in image_groups.items():
        # Create a batch prompt for all objects in this image
        objects_text = ""
        for i, obj in enumerate(objects):
            objects_text += f"Object {i+1} (Step1 label: {obj['step1_label']}):\n{obj['description']}\n\n"
        
        prompt = f"""Based on the following descriptions of objects detected in a single image, classify each object into one of these categories: {', '.join(allowed_classes)}.
        
        {objects_text}
        Return a JSON array where each element corresponds to one of the objects above, in the same order. Each element should be a JSON object with the following structure:
        {{
            "class": "The most appropriate class from the allowed list",
            "confidence": A number between 0 and 1 indicating your confidence,
            "reasoning": "Brief explanation for your classification"
        }}
        
        Only return the JSON array, nothing else."""
        
        # Process each object individually instead of as a batch
        processed_results = []
        
        for i, obj in enumerate(objects):
            # Create a prompt for this specific object
            object_prompt = f"Based on the following description of an object detected in an image, classify it into one of these categories: {', '.join(allowed_classes)}.\n\nObject (Step1 label: {obj['step1_label']}):\n{obj['description']}\n\nReturn a JSON object with the following structure:\n{{\n    \"class\": \"The most appropriate class from the allowed list\",\n    \"confidence\": A number between 0 and 1 indicating your confidence,\n    \"reasoning\": \"Brief explanation for your classification\"\n}}\n\nOnly return the JSON object, nothing else."
            
            # Use the text-only interface for this object
            response = process_with_ollama_text(object_prompt, ollama_model, format_json=True, stream=stream, stream_callback=stream_callback)
            
            if response["success"]:
                response_text = response["response"]
                
                # Try to parse the response as a JSON object
                try:
                    # First try direct JSON parsing
                    try:
                        result = json.loads(response_text)
                        
                        # Validate required fields
                        if not all(k in result for k in ["class", "confidence", "reasoning"]):
                            raise ValueError(f"Missing required fields in response: {response_text}")
                            
                        # Add to processed results
                        processed_results.append(result)
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        # If direct parsing fails, try to extract JSON from text
                        json_obj_start = response_text.find('{')
                        json_obj_end = response_text.rfind('}')
                        
                        if json_obj_start >= 0 and json_obj_end >= 0 and json_obj_end > json_obj_start:
                            try:
                                json_str = response_text[json_obj_start:json_obj_end+1]
                                result = json.loads(json_str)
                                
                                # Validate required fields
                                if not all(k in result for k in ["class", "confidence", "reasoning"]):
                                    raise ValueError(f"Missing required fields in extracted JSON: {json_str}")
                                    
                                # Add to processed results
                                processed_results.append(result)
                            except (json.JSONDecodeError, ValueError) as inner_e:
                                # Create default result if JSON parsing fails
                                default_result = {
                                    "class": "Other",
                                    "confidence": 0.5,
                                    "reasoning": f"Failed to parse valid JSON from Ollama response: {str(inner_e)}"
                                }
                                processed_results.append(default_result)
                        else:
                            # Create default result if no JSON object found
                            default_result = {
                                "class": "Other",
                                "confidence": 0.5,
                                "reasoning": f"No valid JSON object found in response: {response_text}"
                            }
                            processed_results.append(default_result)
                            
                except Exception as e:
                    # Create default result if any other error occurs
                    default_result = {
                        "class": "Other",
                        "confidence": 0.5,
                        "reasoning": f"Error processing Ollama response: {str(e)}"
                    }
                    processed_results.append(default_result)
            else:
                # Handle API error
                error_msg = response.get("error", "Unknown Ollama API error")
                default_result = {
                    "class": "Other",
                    "confidence": 0.5,
                    "reasoning": error_msg
                }
                processed_results.append(default_result)
                
        # Map results back to original indices
        for i, obj in enumerate(objects):
            if i < len(processed_results):
                result = processed_results[i]
                
                # Ensure the class is in the allowed list
                if result["class"] not in allowed_classes:
                    closest_match = min(allowed_classes, key=lambda x: abs(len(x) - len(result["class"])))
                    print(f"Warning: Class '{result['class']}' not in allowed list. Using '{closest_match}' instead.")
                    result["class"] = closest_match
                
                # Ensure confidence is a float between 0 and 1
                try:
                    confidence = float(result["confidence"])
                    result["confidence"] = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    result["confidence"] = 0.7  # Default confidence
            else:
                # Create default result if we don't have enough results
                result = {
                    "class": "Other",
                    "confidence": 0.5,
                    "reasoning": "No result provided by Ollama"
                }
            
            # Store result at the original index
            while len(results) <= obj["index"]:
                results.append(None)
            results[obj["index"]] = result
    
    return results

# Option1 in command line: 
# (py312) lkk@lkk-intel13:~/Developer/VisionLangAnnotate$ ollama run llava
# >>> <image:./example.jpg>
# ... Describe what's happening in this image.

def py_restapi_vlmodel():
    import base64
    import requests

    image_path='/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/bus.jpg'
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": "Describe the scene in the image.",
                "images": [image_base64],
                "stream": False
            },
            timeout=200
        )

    print(response.json()["response"])

def py_chatapi_vlmodel(image_path):
    # Step 1: Load image using PIL
    #image_path='/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/bus.jpg'
    image = Image.open(image_path).convert("RGB")

    # Step 2: Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Step 3: Prepare Ollama vision model request (e.g., llava)
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llava",  # or other supported vision model
        "messages": [
            {
                "role": "user",
                "content": "What is shown in this image?",
                "images": [image_base64]
            }
        ],
        "stream": False
    }

    # Step 4: Send request
    response = requests.post(url, json=payload)
    data = response.json()

    # Step 5: Print the result
    print("Model response:")
    print(data.get("message", {}).get("content", data))

import base64
import requests
from PIL import Image
from io import BytesIO
if __name__ == "__main__":
    image_path = "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/bus.jpg"  # replace with your image path

    #py_restapi_vlmodel()
    py_chatapi_vlmodel(image_path=image_path)

    client = OllamaClient(api_url=DEFAULT_OLLAMA_API_URL)
    result = client.process_text(prompt='What is base64 encoding?', model='llama3.2', format_json=False, system_prompt='You are a helpful assistant and always output result based on evidence.')

    # Example 1: Process a single image with vision model
    image = Image.open(image_path).convert("RGB")
    result2 = client._process_single_vision(image=image, prompt='What is shown in this image?', model='qwen2.5vl')
    print("Single image result:")
    print(result2['response'])
    
    # Example 2: Process multiple images with process_with_ollama_vision
    # Load multiple images
    image_paths = [
        "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/bus.jpg", 
        "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/064224_000100original.jpg",
        "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/sjsupeople.jpg"
    ]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # Define prompts for each image
    prompts = [
        "Describe what you see in this image.",
        "What objects are visible in this image?",
        "Describe what you see in this image.",
    ]
    
    # Process multiple images individually
    multi_result = process_with_ollama_vision(
        images=images,
        prompts=prompts,
        ollama_model="qwen2.5vl"
    )
    
    print("\nMultiple images processed individually:")
    if multi_result["success"]:
        for i, response in enumerate(multi_result["responses"]):
            print(f"\nImage {i+1} response:")
            print(response)
    else:
        print(f"Error: {multi_result.get('error', 'Unknown error')}")
    
    # Example 3: Process multiple images with a combined prompt
    combined_prompt = "Compare these images and tell me what's similar and different between them."
    
    # Process multiple images with a combined prompt
    batch_result = process_with_ollama_vision(
        images=images,
        prompts=["" for _ in images],  # Empty prompts since we're using combined_prompt
        ollama_model="qwen2.5vl",
        combined_prompt=combined_prompt
    )
    
    print("\nMultiple images processed with combined prompt:")
    if batch_result["success"]:
        print(batch_result["response"])
    else:
        print(f"Error: {batch_result.get('error', 'Unknown error')}")
    
    # Example 4: Streaming text response
    def stream_callback(chunk):
        #print(f"Received chunk: {chunk}", end="", flush=True)
        print(f"{chunk}", end="", flush=True)
    
    print("\nStreaming text response:")
    stream_result = process_with_ollama_text(
        prompt="Explain quantum computing in simple terms, step by step.",
        ollama_model="llama3.2",
        format_json=False,
        stream=True,
        stream_callback=stream_callback
    )
    
    print("\n\nFinal streaming result success:", stream_result["success"])
    
    # Example 5: Streaming vision response
    print("\nStreaming vision response:")
    stream_vision_result = process_with_ollama_vision(
        images=[image],  # Just use the first image
        prompts=["Describe this image in detail, mentioning all visible elements."],
        ollama_model="qwen2.5vl",
        stream=True,
        stream_callback=stream_callback
    )
    
    print("\n\nFinal streaming vision result success:", stream_vision_result["success"])
