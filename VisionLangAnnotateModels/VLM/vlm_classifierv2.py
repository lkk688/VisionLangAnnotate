from abc import ABC, abstractmethod
import torch
from PIL import Image, ImageDraw
#from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import List, Tuple, Dict, Any, Optional
import time
import json
import os

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
    """Hugging Face-based VLM backend."""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        # Store the dtype for consistent tensor conversion
        self.dtype = torch.float16 if device == "cuda" else torch.float32
    
    def generate(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        batch_inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Convert input tensors to the same dtype as the model
        batch_inputs = {k: v.to(device=self.device, dtype=self.dtype) 
                       if torch.is_floating_point(v) else v.to(self.device)
                       for k, v in batch_inputs.items()}
        
        with torch.no_grad():
            output = self.model.generate(**batch_inputs, max_new_tokens=50)
        
        return [self.processor.decode(out, skip_special_tokens=True) for out in output]
    
    def get_name(self) -> str:
        return f"HuggingFace-{self.model_name.split('/')[-1]}"

class OpenAIVLM(VLMBackend):
    """OpenAI-based VLM backend (GPT-4V)."""
    
    def __init__(self, api_key: str):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        results = []
        for image, prompt in zip(images, prompts):
            # Convert PIL Image to bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
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


# Example usage
if __name__ == "__main__":
    #testBlip2()
    testBlip2_module()
    # Test with a sample urban scene
    test_vlm_comparison(
        image_path="output/gcs_sources/Sweeper 19303/20250224/064224_000100.jpg", #"sampledata/bus.jpg",
        bbox=None, #(100, 150, 300, 400),  # Example bbox
        prompt="Analyze the vehicle in the image. Is it abandoned, burned, blocking a bike lane, fire hydrant, or red curb? Is it missing parts like tires or windows, or up on jacks?",
        save_dir="./results"
    )