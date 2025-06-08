def extract_labels_from_response(response_text, label_keywords):
    """
    Convert VLM free-form response to structured labels using keyword matching.

    Args:
        response_text (str): Output text from VLM.
        label_keywords (dict): Mapping of label -> trigger keywords.

    Returns:
        list of matched labels
    """
    response = response_text.lower()
    found_labels = []
    for label, keywords in label_keywords.items():
        if any(kw in response for kw in keywords):
            found_labels.append(label)
    return found_labels


# Example predefined label dictionary per super-category
default_label_keywords = {
    "potholes": ["pothole", "potholes"],
    "downed bollard": ["downed bollard", "fallen bollard"],
    "down power line": ["downed power line", "fallen power line", "power cable"],
    "flooding": ["flood", "flooding", "standing water", "sewage"],
    "faded bike lane": ["faded bike lane", "worn bike lane", "unclear markings"],
    "broken street sign": ["broken street sign", "damaged sign", "missing sign"],
    "broken traffic light": ["broken traffic light", "malfunctioning signal", "non-working light"],
    "vehicle blocking bike lane": ["blocking bike lane", "parked in bike lane"],
    "vehicle blocking fire hydrant": ["blocking fire hydrant"],
    "vehicle blocking red curb": ["blocking red curb"],
    "dumped trash": ["dumped trash", "illegal dumping"],
    "yard waste": ["yard waste", "leaf pile", "branch pile"],
    "glass": ["glass shards", "broken glass"],
    "residential trash cans in the bike lane": ["residential trash can in bike lane"],
    "commercial dumpsters in the bike lane": ["commercial dumpster in bike lane"],
    "graffiti": ["graffiti", "spray paint"],
    "construction sign": ["construction sign"],
    "cone": ["construction cone", "orange cone"],
    "blocked road": ["blocked road", "closed road"],
    "streetlight outage": ["streetlight outage", "light is off"],
    "tree overhang": ["tree overhang", "overhanging branch"],
    "burned vehicle": ["burned vehicle"],
    "vehicle on jacks": ["vehicle on jacks", "vehicle on blocks"],
    "shattered windows": ["shattered window", "broken window"],
    "missing tires": ["missing tire", "no wheel"],
    "street vendors in bike lane": ["vendor in bike lane", "selling in bike lane"]
}


# llm_label_extractor.py

import openai

# Example: label set per prompt category
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

def extract_labels_with_llm(vlm_response, category, model="gpt-3.5-turbo"):
    label_list = LABEL_SETS.get(category, [])
    label_string = ", ".join(f'"{label}"' for label in label_list)

    prompt = f"""
Given the following VLM response:
\"{vlm_response.strip()}\"

Extract all applicable labels from the following list:
[{label_string}]

Return a Python list of the matching labels.
"""

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        output = response.choices[0].message["content"]
        labels = eval(output) if output.startswith("[") else []
        return [label.strip() for label in labels if label.strip() in label_list]
    except Exception as e:
        print("LLM extraction error:", e)
        return []