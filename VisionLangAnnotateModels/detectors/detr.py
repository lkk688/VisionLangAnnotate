# detectors/detr.py

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

class DETRDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", threshold=0.3):
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
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
            box = box.tolist()  # [x0, y0, x1, y1]
            label_str = self.model.config.id2label[label.item()].lower()
            detections.append({
                "bbox": box,
                "label": label_str,
                "confidence": score.item()
            })
        return detections