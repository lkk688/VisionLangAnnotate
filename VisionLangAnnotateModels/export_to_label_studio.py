import json
import os
from PIL import Image

def convert_to_label_studio(results, image_path, task_id="auto_1"):
    image = Image.open(image_path)
    width, height = image.size

    annotations = []
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        label = r["vlm_description"]
        region = {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "x": x1 / width * 100,
                "y": y1 / height * 100,
                "width": (x2 - x1) / width * 100,
                "height": (y2 - y1) / height * 100,
                "rectanglelabels": [label]
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels"
        }
        annotations.append(region)

    data = {
        "id": task_id,
        "data": {
            "image": f"/data/local-files/?d={os.path.basename(image_path)}"
        },
        "predictions": [
            {
                "result": annotations,
                "model_version": "two-step-vlm-v1"
            }
        ]
    }

    return data


def export_batch(results_list, image_paths, output_file="label_studio_output.json"):
    dataset = []
    for results, path in zip(results_list, image_paths):
        data = convert_to_label_studio(results, path)
        dataset.append(data)

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSaved Label Studio export to: {output_file}")


# Run pipeline + export
from pipeline import run_pipeline
from detectors.yolov8 import YOLOv8Detector
from detectors.detr import DETRDetector
from detectors.rtdetr import RTDETRDetector
from export_to_label_studio import export_batch

images = ["data/street_001.jpg", "data/street_002.jpg"]

detectors = [
    YOLOv8Detector(),
    DETRDetector(),
    RTDETRDetector()
]

all_results = [run_pipeline(img, detectors) for img in images]
export_batch(all_results, images)