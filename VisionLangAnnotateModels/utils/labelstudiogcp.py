import os
import json
import matplotlib.pyplot as plt
import numpy as np
#pip install --upgrade google-cloud-storage
from google.cloud import storage
from PIL import Image

# ---------------- CONFIGURATION ----------------
ANNOTATION_BUCKET_NAME = "roadsafetytarget"
SOURCE_IMAGE_BUCKET_NAME = "roadsafetysource"
ANNOTATION_FOLDER = ""  # Leave empty if annotations are in root
LOCAL_DOWNLOAD_DIR = "./output/gcp_downloads"
LOCAL_VISUALIZED_DIR = "./output/labelstuido_visualized"
os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(LOCAL_VISUALIZED_DIR, exist_ok=True)

# ---------------- GCS SETUP ----------------
#gcloud auth application-default login
storage_client = storage.Client()
annotation_bucket = storage_client.bucket(ANNOTATION_BUCKET_NAME)
source_image_bucket = storage_client.bucket(SOURCE_IMAGE_BUCKET_NAME)

# ---------------- DOWNLOAD ANNOTATIONS ----------------
def download_annotation_files():
    annotations = {}
    blobs = list(annotation_bucket.list_blobs(prefix=ANNOTATION_FOLDER))

    for blob in blobs:
        filename = os.path.basename(blob.name)
        try:
            print(f"Downloading annotation: {filename}")
            data = blob.download_as_bytes()
            parsed_json = json.loads(data)
            annotations[filename] = parsed_json
        except Exception as e:
            print(f"Skipping {filename}: invalid JSON - {e}")
    
    return annotations

# ---------------- DOWNLOAD IMAGES (preserve structure) ----------------
def download_image_from_gcs(gcs_path):
    # gcs_path: 'gs://roadsafetysource/Sweeper 19303/folder/image.jpg'
    path = gcs_path.replace("gs://", "")
    bucket_name, *object_parts = path.split("/")
    object_name = "/".join(object_parts)

    local_image_path = os.path.join(LOCAL_DOWNLOAD_DIR, object_name)
    local_image_dir = os.path.dirname(local_image_path)
    os.makedirs(local_image_dir, exist_ok=True)

    blob = source_image_bucket.blob(object_name)
    if not blob.exists():
        raise FileNotFoundError(f"Image not found in GCS: {object_name}")
    
    blob.download_to_filename(local_image_path)
    return local_image_path, object_name  # return GCS-relative path


# ---------------- VISUALIZATION ----------------
def visualize(image_path, annotation_data):
    image = np.array(Image.open(image_path).convert("RGB"))
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for ann in annotation_data.get("annotations", []):
        for item in ann["result"]:
            if item["type"] == "rectanglelabels":
                val = item["value"]
                x, y, w, h = val["x"], val["y"], val["width"], val["height"]
                label = val.get("rectanglelabels", [""])[0]
                img_h, img_w = image.shape[:2]

                rect_x = x / 100 * img_w
                rect_y = y / 100 * img_h
                rect_w = w / 100 * img_w
                rect_h = h / 100 * img_h

                rect = plt.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                     edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                ax.text(rect_x, rect_y - 5, label, color='red', fontsize=12, backgroundcolor='white')

    plt.axis('off')
    plt.show()

def visualize_and_save(image_path, annotation_data, relative_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for ann in annotation_data.get("annotations", []):
        for item in ann["result"]:
            if item["type"] == "rectanglelabels":
                val = item["value"]
                x, y, w, h = val["x"], val["y"], val["width"], val["height"]
                label = val.get("rectanglelabels", [""])[0]
                img_h, img_w = image.shape[:2]

                rect_x = x / 100 * img_w
                rect_y = y / 100 * img_h
                rect_w = w / 100 * img_w
                rect_h = h / 100 * img_h

                rect = plt.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                     edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                ax.text(rect_x, rect_y - 5, label, color='red', fontsize=12, backgroundcolor='white')

    ax.axis('off')

    # Save figure to visualized/...
    save_path = os.path.join(LOCAL_VISUALIZED_DIR, relative_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def download_all_images_from_gcs(bucket_name, gcs_prefix="", local_download_dir="downloads"):
    """
    Download all image files from a GCS bucket, preserving the original folder structure.

    Args:
        bucket_name (str): Name of the GCS bucket.
        gcs_prefix (str): Optional prefix/folder path inside the bucket.
        local_download_dir (str): Local folder where images will be saved.

    Returns:
        List of downloaded local file paths.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_prefix)

    downloaded_files = []

    for blob in blobs:
        if blob.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
            # Build local path by appending blob.name to local_download_dir
            local_path = os.path.join(local_download_dir, blob.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            downloaded_files.append(local_path)
            print(f"✅ Downloaded: {blob.name} → {local_path}")

    return downloaded_files

# ---------------- MAIN FLOW ----------------
def main(SNAPSHOT_JSON_PATH):
    # Load full snapshot file
    with open(SNAPSHOT_JSON_PATH, "r") as f:
        snapshot_data = json.load(f)

    for entry in snapshot_data:
        try:
            # Skip if there are no annotations
            if not entry.get("annotations"):
                continue

            # Further check if at least one result exists
            has_results = any("result" in ann and ann["result"] for ann in entry["annotations"])
            if not has_results:
                continue

            image_gcs_path = entry["data"]["image"]
            annotations = {"annotations": entry["annotations"]}

            local_path, relative_path = download_image_from_gcs(image_gcs_path)
            visualize_and_save(local_path, annotations, relative_path)
            print(f"✅ Visualized and saved: {relative_path}")

        except Exception as e:
            print(f"❌ Failed to process entry {entry.get('id', 'unknown')}: {e}")

if __name__ == "__main__":
    #annotations = download_annotation_files() #1664
    # Download all images from gs://roadsafetysource/Sweeper 19303/
    downloaded = download_all_images_from_gcs(
        bucket_name="roadsafetysource",
        gcs_prefix="",
        local_download_dir="./output/gcs_sources"
    )
    #main(SNAPSHOT_JSON_PATH="output/export_146842_project-146842-at-2025-06-08-23-56-46ec1ea2.json")