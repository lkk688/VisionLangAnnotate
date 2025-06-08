from PIL import Image
import os

from detectors.yolov8 import YOLOv8Detector
from detectors.detr import DETRDetector
from detectors.rtdetr import RTDETRDetector
from detectors.ensemble import ensemble_detections
from vlm_classifier import VLMClassifier
from prompts import prompts


def crop_image(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image.crop((x1, y1, x2, y2))


def run_pipeline(image_path, selected_detectors, device="cuda"):
    image = Image.open(image_path).convert("RGB")
    detection_results = []

    # Run all selected detectors
    for detector in selected_detectors:
        detection_results.append(detector.detect(image))

    # Ensemble outputs from different detectors
    detections = ensemble_detections(detection_results)

    if not detections:
        print("No objects detected.")
        return []

    vlm = VLMClassifier(device=device)

    # Prepare crops + batched prompts
    image_crops_with_prompts = []
    metadata = []

    for det in detections:
        bbox = det["bbox"]
        label = det["label"]
        crop = crop_image(image, bbox)

        # Match to prompt key
        prompt_key = next((key for key in prompts if key in label.lower()), "other")
        prompt = prompts[prompt_key]

        image_crops_with_prompts.append((crop, prompt))
        metadata.append({
            "bbox": bbox,
            "step1_label": label,
            "prompt_category": prompt_key
        })

    # VLM inference
    answers = vlm.classify(image_crops_with_prompts)

    # Combine predictions
    results = []
    for meta, answer in zip(metadata, answers):
        result = {
            "bbox": meta["bbox"],
            "step1_label": meta["step1_label"],
            "prompt_category": meta["prompt_category"],
            "vlm_description": answer
        }
        results.append(result)

    return results

#Instead of cropping, we highlight detected objects in full scene images.
def run_pipeline_overlay(image_path, selected_detectors, device="cuda"):
    image = Image.open(image_path).convert("RGB")
    detection_results = []

    # Run all selected detectors
    for detector in selected_detectors:
        detection_results.append(detector.detect(image))

    # Ensemble outputs from different detectors
    detections = ensemble_detections(detection_results)

    if not detections:
        print("No objects detected.")
        return []

    vlm = VLMClassifier(device=device)

    # Prepare full context images with bounding boxes and prompts
    full_context_images_with_bbox = []
    prompt_list = []
    metadata = []

    for det in detections:
        bbox = det["bbox"]
        label = det["label"]

        # Draw the box on the original image
        boxed_image = vlm.embed_box_on_image(image, bbox)

        # Match to prompt key
        prompt_key = next((key for key in prompts if key in label.lower()), "other")
        prompt = prompts[prompt_key]

        full_context_images_with_bbox.append(boxed_image)
        prompt_list.append(prompt)
        metadata.append({
            "bbox": bbox,
            "step1_label": label,
            "prompt_category": prompt_key
        })

    # VLM inference using full-image context + spatially informed prompts
    answers = vlm.classify(full_context_images_with_bbox, prompt_list)

    # Combine predictions
    results = []
    for meta, answer in zip(metadata, answers):
        result = {
            "bbox": meta["bbox"],
            "step1_label": meta["step1_label"],
            "prompt_category": meta["prompt_category"],
            "vlm_description": answer
        }
        results.append(result)

    return results

def run_pipeline_twoview(image_path, selected_detectors, device="cuda"):
    image = Image.open(image_path).convert("RGB")
    detection_results = []

    # Run all selected detectors
    for detector in selected_detectors:
        detection_results.append(detector.detect(image))

    # Ensemble outputs from different detectors
    detections = ensemble_detections(detection_results)

    if not detections:
        print("No objects detected.")
        return []

    vlm = VLMClassifier(device=device)

    image_pairs = []  # (crop + context)
    prompt_list = []
    metadata = []

    for det in detections:
        bbox = det["bbox"]
        label = det["label"]

        # Positional description
        position = describe_position(bbox, image.size)

        # Prompt key
        prompt_key = next((key for key in prompts if key in label.lower()), "other")
        base_prompt = prompts[prompt_key]

        # Contextual image with overlay
        context_img = vlm.embed_box_on_image(image, bbox)

        # Crop for two-view
        x1, y1, x2, y2 = map(int, bbox)
        crop = image.crop((x1, y1, x2, y2))
        pair_image = vlm.two_view_image(crop, context_img)

        # Nearby object relations
        relations = detect_relations(bbox, detections)
        rel_info = f" Nearby objects: {', '.join(set(relations))}." if relations else ""

        # Final prompt
        prompt = (
            f"The object is in the {position} of the image. {base_prompt.strip()} {rel_info.strip()}"
        )

        image_pairs.append(pair_image)
        prompt_list.append(prompt)
        metadata.append({
            "bbox": bbox,
            "step1_label": label,
            "prompt_category": prompt_key
        })

    answers = vlm.classify(image_pairs, prompt_list)

    results = []
    for meta, answer in zip(metadata, answers):
        results.append({
            "bbox": meta["bbox"],
            "step1_label": meta["step1_label"],
            "prompt_category": meta["prompt_category"],
            "vlm_description": answer
        })

    return results

def run_pipeline_structured(image_path, selected_detectors, device="cuda"):
    image = Image.open(image_path).convert("RGB")
    detection_results = []

    # Run all selected detectors
    for detector in selected_detectors:
        detection_results.append(detector.detect(image))

    # Ensemble outputs from different detectors
    detections = ensemble_detections(detection_results)

    if not detections:
        print("No objects detected.")
        return []

    vlm = VLMClassifier(device=device)

    # Group all detections under shared structured prompt
    categories_in_image = set(det["label"].split()[0] for det in detections)
    prompt, label_pool = build_multi_category_prompt(categories_in_image)

    # Draw all boxes on image for global context
    full_image_context = image.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(full_image_context)
    for det in detections:
        draw.rectangle(det["bbox"], outline="red", width=2)

    # Classify once
    answers = vlm.classify([full_image_context], [prompt])
    response = answers[0] if answers else ""
    matched_labels = parse_multi_label_response(response, label_pool)

    # Assign labels to bounding boxes if matched
    results = []
    for det in detections:
        results.append({
            "bbox": det["bbox"],
            "step1_label": det["label"],
            "vlm_description": response,
            "vlm_labels": matched_labels
        })

    return results

def map_labels_to_boxes(vlm_labels, detections):
    label_to_box_map = []
    label_keywords = {
        "potholes": ["road", "pavement"],
        "flooding": ["road", "pavement"],
        "faded bike lane": ["road", "lane", "stripe"],
        "downed bollard": ["bollard", "pole"],
        "down power line": ["wire", "line", "cable"],
        "dumped trash": ["trash", "debris"],
        "yard waste": ["tree", "branch", "leaves"],
        "graffiti": ["wall", "sign", "surface"],
        "broken street sign": ["sign"],
        "vehicle blocking bike lane": ["vehicle", "car", "truck"],
        "tree overhang": ["tree", "branch"],
        # extend as needed
    }

    for label in vlm_labels:
        matched = False
        for det in detections:
            if any(kw in det["step1_label"].lower() for kw in label_keywords.get(label, [])):
                label_to_box_map.append({
                    "label": label,
                    "bbox": det["bbox"],
                    "step1_label": det["step1_label"]
                })
                matched = True
                break
        if not matched:
            label_to_box_map.append({
                "label": label,
                "bbox": None,
                "step1_label": None
            })
    return label_to_box_map
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    # Choose detectors
    detectors = [
        YOLOv8Detector(model_path="yolov8x.pt"),
        DETRDetector(model_name="facebook/detr-resnet-50"),
        RTDETRDetector(model_name="SenseTime/deformable-detr")
    ]

    output = run_pipeline(args.image, detectors, device=args.device)

    print("\nDetection + VLM Classification Results:")
    for obj in output:
        print(obj)