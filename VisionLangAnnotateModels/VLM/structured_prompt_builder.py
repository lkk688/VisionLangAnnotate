LABEL_SETS = {
    "road_surface": [
        "potholes", "downed bollard", "down power line", "flooding", "faded bike lane"
    ],
    "road_sign": [
        "broken street sign"
    ],
    "traffic_light": [
        "broken traffic light"
    ],
    "vehicle": [
        "vehicle blocking bike lane", "vehicle blocking fire hydrant", "vehicle blocking red curb",
        "burned vehicle", "vehicle on jacks", "shattered windows", "missing tires"
    ],
    "trash": [
        "dumped trash", "yard waste", "glass"
    ],
    "trash_can": [
        "residential trash cans in the bike lane", "commercial dumpsters in the bike lane"
    ],
    "tree": [
        "tree overhang"
    ],
    "other": [
        "graffiti", "construction sign", "cone", "blocked road", "street vendors in bike lane"
    ],
    "light_pole": [
        "streetlight outage"
    ]
}

def build_structured_prompt(category):
    labels = LABEL_SETS.get(category, [])
    label_list = ", ".join(f'"{label}"' for label in labels)
    return (
        f"From the following list of labels: [{label_list}], "
        "which ones are visible in the image? "
        "Respond with a Python list of the matching labels only."
    )

def parse_label_list_response(response, label_set):
    try:
        labels = eval(response.strip())
        return [label for label in labels if label in label_set]
    except Exception as e:
        print("Invalid response format:", e)
        return []