from shapely.geometry import box as shapely_box

def detect_relations(target_box, all_boxes_with_labels, iou_thresh=0.1, dist_thresh=50):
    context = []
    target_shape = shapely_box(*target_box)

    for other in all_boxes_with_labels:
        if other["bbox"] == target_box:
            continue
        other_shape = shapely_box(*other["bbox"])
        iou = target_shape.intersection(other_shape).area / target_shape.union(other_shape).area
        dist = target_shape.distance(other_shape)

        if iou > iou_thresh or dist < dist_thresh:
            context.append(other["label"].lower())
    return context