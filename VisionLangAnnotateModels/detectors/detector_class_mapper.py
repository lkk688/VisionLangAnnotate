COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

step1_classes = [
    "person",
    "vehicle",
    "train",
    "mobility_device",
    "graffiti",
    "trash/waste"
    "traffic infrastructure",
    "pothole",
    "pole/utility"
    "vegetation",
    "construction",
    "street furniture"
    "animal",
    "other"
]

step2_classes = {
    "person": [
        "pedestrian",
        "street vendor",
        "worker in uniform",
        "person on wheelchair",
        "person pushing stroller",
        "person crossing street",
        "child near road",
        "homeless person sleeping"
    ],
    "vehicle": [
        "car",
        "van",
        "truck",
        "bus",
        "motorcycle",
        "ambulance",
        "fire truck",
        "police car",
        "vehicle blocking bike lane",
        "vehicle blocking red curb",
        "vehicle blocking fire hydrant",
        "burned vehicle",
        "vehicle on jacks or blocks",
        "abandoned vehicle",
        "vehicle with shattered windows",
        "delivery truck parked illegally"
    ],
    "train": [
        "commuter train",
        "freight train",
        "train at station",
        "light rail vehicle",
        "tram"
    ],
    "mobility_device": [
        "bicycle",
        "shared bike",
        "scooter",
        "shared e-scooter",
        "wheelchair",
        "motorized wheelchair",
        "baby stroller",
        "skateboard"
    ],
    "graffiti": [
        "graffiti on wall",
        "graffiti on public sign",
        "graffiti on pole or utility box",
        "graffiti on vehicle"
    ],
    "trash/waste": [
        "dumped trash bag",
        "overfilled trash bin",
        "yard waste pile",
        "glass debris on road",
        "commercial dumpster in bike lane",
        "residential trash can on sidewalk",
        "illegal dumping",
        "mattress or furniture dumped on street"
    ],
    "traffic infrastructure": [
        "damaged street sign",
        "broken traffic light",
        "faded bike lane paint",
        "construction sign in lane",
        "streetlight outage",
        "missing road markings"
    ],
    "pothole": [
        "small pothole",
        "deep pothole",
        "pothole with water",
        "repaired pothole",
        "cracked asphalt"
    ],
    "pole/utility": [
        "utility pole",
        "leaning pole",
        "bollard knocked over",
        "exposed cable or wire",
        "downed power line",
        "light pole",
        "street signal post",
        "cable box or telecom unit"
    ],
    "vegetation": [
        "tree overhanging sidewalk",
        "fallen branch on street",
        "overgrown bushes",
        "vegetation blocking sign",
        "planter on sidewalk"
    ],
    "construction": [
        "construction equipment on sidewalk",
        "construction cones in road",
        "temporary fencing",
        "scaffolding blocking path",
        "roadwork zone",
        "sidewalk under repair"
    ],
    "street furniture": [
        "public bench",
        "bus stop shelter",
        "information kiosk",
        "trash can",
        "bike rack",
        "newspaper stand"
    ],
    "animal": [
        "stray dog",
        "pigeon flock",
        "cat on sidewalk",
        "dead animal on road",
        "animal waste",
        "wildlife in urban area"
    ],
    "other": [
        "unidentified obstruction",
        "flooded sidewalk",
        "miscellaneous hazard",
        "open manhole",
        "shopping cart left in street",
        "temporary stage or event setup"
    ]
}

# Mapping from COCO class names to custom Step 1 class labels
# COCO_TO_STEP1 = {
#     "car": "vehicle",
#     "truck": "vehicle",
#     "bus": "vehicle",
#     "motorcycle": "vehicle",
#     "bicycle": "vehicle",
#     "person": "person",
#     "traffic light": "traffic_light",
#     "stop sign": "road_sign",
#     "parking meter": "road_sign",
#     "bench": "other",
#     "fire hydrant": "road_sign",
#     "trash can": "trash_can",
#     "dog": "other",
#     "cat": "other",
#     "bird": "other",
#     "potted plant": "other",
#     "chair": "other",
#     "sofa": "other",
#     "tv": "other",
#     "keyboard": "other",
#     "backpack": "other",
#     "umbrella": "other",
#     "handbag": "other",
#     "tie": "other",
#     "suitcase": "other",
#     "frisbee": "other",
#     "skateboard": "other",
#     "surfboard": "other",
#     "bottle": "other",
#     "cup": "other",
#     "fork": "other",
#     "knife": "other",
#     "spoon": "other",
#     "bowl": "other",
#     "banana": "trash",
#     "apple": "trash",
#     "sandwich": "trash",
#     "orange": "trash",
#     "broccoli": "trash",
#     "carrot": "trash",
#     "hot dog": "trash",
#     "pizza": "trash",
#     "donut": "trash",
#     "cake": "trash",
#     "book": "other",
#     "cell phone": "other",
#     "laptop": "other",
#     "mouse": "other",
#     "remote": "other",
#     "microwave": "other",
#     "oven": "other",
#     "toaster": "other",
#     "sink": "other",
#     "refrigerator": "other",
#     "scissors": "other",
#     "teddy bear": "other",
#     "hair drier": "other",
#     "toothbrush": "other"
# }

# def map_coco_label_to_step1(coco_label):
#     return COCO_TO_STEP1.get(coco_label.lower(), "other")

coco_to_step1_map = {
    # People
    "person": "person",

    # Vehicles
    "bicycle": "mobility_device",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "airplane": "other",
    "bus": "vehicle",
    "train": "train",
    "truck": "vehicle",
    "boat": "other",

    # Traffic infrastructure
    "traffic light": "traffic infrastructure",
    "fire hydrant": "traffic infrastructure",
    "stop sign": "traffic infrastructure",
    "parking meter": "traffic infrastructure",
    "bench": "street furniture",

    # Animals
    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",

    # Carry-on objects (default to other or obstruction)
    "backpack": "other",
    "umbrella": "other",
    "handbag": "other",
    "tie": "other",
    "suitcase": "other",

    # Sports
    "frisbee": "other",
    "skis": "other",
    "snowboard": "other",
    "sports ball": "other",
    "kite": "other",
    "baseball bat": "other",
    "baseball glove": "other",
    "skateboard": "mobility_device",
    "surfboard": "other",
    "tennis racket": "other",

    # Dining items (indoor or sidewalk use)
    "bottle": "trash/waste",
    "wine glass": "trash/waste",
    "cup": "trash/waste",
    "fork": "trash/waste",
    "knife": "trash/waste",
    "spoon": "trash/waste",
    "bowl": "trash/waste",
    "banana": "trash/waste",
    "apple": "trash/waste",
    "sandwich": "trash/waste",
    "orange": "trash/waste",
    "broccoli": "trash/waste",
    "carrot": "trash/waste",
    "hot dog": "trash/waste",
    "pizza": "trash/waste",
    "donut": "trash/waste",
    "cake": "trash/waste",

    # Furniture and indoor objects
    "chair": "street furniture",
    "couch": "street furniture",
    "potted plant": "vegetation",
    "bed": "other",
    "dining table": "street furniture",
    "toilet": "trash/waste",
    "tv": "other",
    "laptop": "other",
    "mouse": "other",
    "remote": "other",
    "keyboard": "other",
    "cell phone": "other",
    "microwave": "other",
    "oven": "other",
    "toaster": "other",
    "sink": "other",
    "refrigerator": "other",
    "book": "other",
    "clock": "other",
    "vase": "other",
    "scissors": "other",
    "teddy bear": "other",
    "hair drier": "other",
    "toothbrush": "other"
}

def class_name_mapper(detected_labels, nomatch_placeholder="other"):
    # Process each detected label
    updated_labels = []
    for label in detected_labels:
        newlabel = coco_to_step1_map.get(label, nomatch_placeholder)
        updated_labels.append(newlabel)
    return updated_labels

import torch
from transformers import pipeline
class NLPClass_Mapper:
    def __init__(self, class_list=None):
        self.classifier = None
        # Initialize the zero-shot classification pipeline
        try:
            self.classifier = pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli", 
                                device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            print(f"Error loading classification model: {e}")
        self.coco_names = COCO_CLASSES
        self.class_names = []
        self.cocolabel_map = {}
        if class_list is not None:
            self.class_names, self.cocolabel_map = self.class_names_update(self.coco_names, class_list)
            print("cocolabel map:", self.cocolabel_map)

    def class_names_update(self, detected_labels, interested_labels, nomatch_placeholder="other"):
        """
        Update detected class names to match user's interested labels using a text classification model.
        
        This function uses a zero-shot text classification model to map detected class names
        (which may be from standard datasets like COCO) to the user's interested categories.
        For example, it can map 'person' to 'human', 'car'/'truck' to 'vehicle', etc.
        
        The function uses the Hugging Face transformers library with the facebook/bart-large-mnli
        model for zero-shot classification. It computes similarity scores between each detected
        label and all interested labels, then selects the best match if the confidence score
        exceeds a threshold (0.5).
        
        Args:
            detected_labels: List of detected class names (e.g., COCO classes like 'person', 'car')
            interested_labels: List of user's interested class names (e.g., 'human', 'vehicle')
            
        Returns:
            List of updated class names that match the interested_labels. If no good match is found
            for a particular label, the original label is retained.
            
        Example:
            >>> detected_labels = ['person', 'car', 'dog']
            >>> interested_labels = ['human', 'vehicle', 'animal']
            >>> model.class_names_update(detected_labels, interested_labels)
            ['human', 'vehicle', 'animal']
        """
        # If no interested labels provided, return original labels
        if not interested_labels or len(interested_labels) == 0:
            return detected_labels
            
        updated_labels = []
        label_map = {}
        
        # Process each detected label
        for label in detected_labels:
            # Skip empty labels
            if not label or label.strip() == "":
                if nomatch_placeholder is not None:
                    updated_labels.append(nomatch_placeholder)
                    label_map[label] = nomatch_placeholder
                else:
                    updated_labels.append(label)
                    label_map[label] = label
                continue
                
            try:
                # Use zero-shot classification to find the best match
                result = self.classifier(label, interested_labels, multi_label=False)
                
                # Get the highest scoring match
                best_match_idx = result['scores'].index(max(result['scores']))
                best_match = result['labels'][best_match_idx]
                best_score = result['scores'][best_match_idx]
                
                # Only use the match if the score is above a threshold
                if best_score > 0.5:
                    updated_labels.append(best_match)
                    label_map[label] = best_match
                else:
                    if nomatch_placeholder is not None:
                        updated_labels.append(nomatch_placeholder)
                        label_map[label] = nomatch_placeholder
                    else:
                        # Keep original if no good match
                        updated_labels.append(label)
                        label_map[label] = label
                    
                #print(f"Mapped '{label}' to '{best_match}' with confidence {best_score:.2f}")
                
            except Exception as e:
                print(f"Error classifying label '{label}': {e}")
                updated_labels.append(label)  # Keep original on error
                label_map[label] = label
        
        return updated_labels, label_map

def test_class_names_update():
    # Initialize the model
    model = NLPClass_Mapper(class_list=['human', 'vehicle', 'animal'])
    
    # Test with some sample detected labels and interested labels
    detected_labels = ['person', 'car', 'truck', 'bicycle', 'dog']
    interested_labels = ['human', 'vehicle', 'animal']
    
    print("\nTesting class_names_update function:")
    print(f"Detected labels: {detected_labels}")
    print(f"Interested labels: {interested_labels}")
    
    # Call the function
    updated_labels = model.class_names_update(detected_labels, interested_labels)
    
    print(f"Updated labels: {updated_labels}")
    
    # Test with empty interested labels
    print("\nTesting with empty interested labels:")
    empty_interested = []
    updated_labels = model.class_names_update(detected_labels, empty_interested)
    print(f"Updated labels: {updated_labels}")
    
    # Test with empty detected labels
    print("\nTesting with empty detected labels:")
    empty_detected = []
    updated_labels = model.class_names_update(empty_detected, interested_labels)
    print(f"Updated labels: {updated_labels}")

if __name__ == "__main__":
    test_class_names_update()