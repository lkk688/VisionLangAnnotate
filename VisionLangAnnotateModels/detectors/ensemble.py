import numpy as np
#pip install ensemble-boxes
from ensemble_boxes import nms
from ensemble_boxes import weighted_boxes_fusion

#implements Weighted Boxes Fusion (WBF) to ensemble detection outputs across models, improving consensus over redundant or overlapping boxes.
def wbf_ensemble_detections(detection_lists, iou_thr=0.5, skip_box_thr=0.3):
    boxes_list, scores_list, labels_list = [], [], []
    image_size = 1024  # normalize relative to this size; adjust as needed

    for detections in detection_lists:
        boxes, scores, labels = [], [], []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            boxes.append([x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size])
            scores.append(d["confidence"])
            labels.append(d["label"])
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    if not boxes_list:
        return []

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )

    results = []
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [coord * image_size for coord in b]
        results.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": s,
            "label": l
        })
    return results

def ensemble_detections(detection_lists, iou_thr=0.5):
    boxes, scores, labels = [], [], []
    for dets in detection_lists:
        for d in dets:
            boxes.append(d["bbox"])
            scores.append(d["confidence"])
            labels.append(d["label"])

    # normalize and run NMS
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    boxes = boxes / 1024  # assume input images 1024x1024, adjust as needed
    boxes_list = [boxes.tolist()]
    scores_list = [scores.tolist()]
    labels_list = [[label] for label in labels]

    ensembled_boxes, ensembled_scores, ensembled_labels = nms(
        boxes_list, scores_list, labels_list, iou_thr=iou_thr
    )

    # re-scale
    ensembled_boxes = (np.array(ensembled_boxes) * 1024).tolist()
    return [
        {"bbox": b, "label": l[0], "confidence": s}
        for b, s, l in zip(ensembled_boxes, ensembled_scores, ensembled_labels)
    ]