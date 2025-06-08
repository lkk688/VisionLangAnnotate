def describe_position(bbox, image_size):
    x1, y1, x2, y2 = bbox
    w, h = image_size
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    horizontal = "left" if cx < w / 3 else "center" if cx < 2 * w / 3 else "right"
    vertical = "top" if cy < h / 3 else "middle" if cy < 2 * h / 3 else "bottom"
    return f"{vertical}-{horizontal}"