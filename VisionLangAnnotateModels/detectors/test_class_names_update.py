import torch
import sys
sys.path.append('/home/lkk/Developer/VisionLangAnnotate')
from VisionLangAnnotateModels.detectors.inference import ModelInference

# Create a test case for class_names_update function
def test_class_names_update():
    # Initialize the model
    model = ModelInference(model_type='yolo', model_name="yolov8n.pt", device='cpu')
    
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