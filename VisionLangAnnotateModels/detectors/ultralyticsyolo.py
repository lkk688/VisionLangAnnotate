#pip install ultralytics
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        ## Load a pretrained YOLO11n model
        #model = YOLO("yolo11n.pt")

    def detect(self, image):
        results = self.model(image)
        #results[0].show()  # Display results
        detections = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                cls = int(r.boxes.cls[0].item())
                label = self.model.names[cls]
                conf = r.boxes.conf[0].item()
                detections.append({"bbox": box.tolist(), "label": label, "confidence": conf})
        return detections