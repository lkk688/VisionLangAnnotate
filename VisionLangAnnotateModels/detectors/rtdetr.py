# detectors/rtdetr.py

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForObjectDetection

class RTDETRDetector:
    def __init__(self, model_name="SenseTime/deformable-detr", threshold=0.3):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model.eval()
        self.threshold = threshold

    def detect(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.threshold)[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_str = self.model.config.id2label[label.item()].lower()
            detections.append({
                "bbox": box.tolist(),
                "label": label_str,
                "confidence": score.item()
            })
        return detections