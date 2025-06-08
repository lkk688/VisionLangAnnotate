from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class VLMClassifier:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.device = device

    def embed_box_on_image(self, image, bbox, color="red"):
        """Draw bounding box on image for visual context."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=color, width=4)
        return img

    #supports multi-image input or combine images into a visual grid for BLIP.
    #Horizontally concatenate crop + context
    def two_view_image(self, crop, full_context):
        context_resized = full_context.resize(crop.size)
        combined = Image.new("RGB", (crop.width * 2, crop.height))
        combined.paste(crop, (0, 0))
        combined.paste(context_resized, (crop.width, 0))
        return combined

    def classify(self, image_crops_with_prompts):
        results = []
        for img_crop, prompt in image_crops_with_prompts:
            inputs = self.processor(img_crop, prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs)
            answer = self.processor.decode(output[0], skip_special_tokens=True)
            results.append(answer)
        return results
    
    def classify_relativepos(self, object_crops, full_context_images, prompts):
        results = []
        for crop, context_img, prompt in zip(object_crops, full_context_images, prompts):
            # Combine prompt with relative position
            context_prompt = prompt + " The object is marked in the full image."
            inputs = self.processor(images=context_img, text=context_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50)
            answer = self.processor.decode(output[0], skip_special_tokens=True)
            results.append(answer)
        return results
    
    #full image with bounding box overlay, preserves spatial context
    def classify_overlay(self, full_context_images_with_bbox, prompts):
        results = []
        for context_img, prompt in zip(full_context_images_with_bbox, prompts):
            full_prompt = prompt + " The object is highlighted in the image with a red box."
            inputs = self.processor(images=context_img, text=full_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50)
            answer = self.processor.decode(output[0], skip_special_tokens=True)
            results.append(answer)
        return results
    
    def classify_twoview(self, image_pairs, prompts):
        results = []
        batch_inputs = self.processor(
            images=image_pairs,
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**batch_inputs, max_new_tokens=50)

        for out in output:
            answer = self.processor.decode(out, skip_special_tokens=True)
            results.append(answer)
        return results

    def classify_batch(self, full_context_images_with_bbox, prompts):
        results = []

        batch_inputs = self.processor(
            images=full_context_images_with_bbox,
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**batch_inputs, max_new_tokens=50)

        for out in output:
            answer = self.processor.decode(out, skip_special_tokens=True)
            results.append(answer)

        return results