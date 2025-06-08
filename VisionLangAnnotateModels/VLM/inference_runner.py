import os
import cv2
import tempfile
import shutil
from glob import glob
from pipeline import run_pipeline
from detectors.yolov8 import YOLOv8Detector
from detectors.detr import DETRDetector
from detectors.rtdetr import RTDETRDetector
from export_to_label_studio import export_batch
from google.cloud import storage


def setup_detectors():
    return [
        YOLOv8Detector(),
        DETRDetector(),
        RTDETRDetector()
    ]


def infer_from_video(video_path, output_dir, sample_rate=30):
    print(f"\nExtracting frames from {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if count % sample_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"Saved {saved} frames to {output_dir}")
    return sorted(glob(os.path.join(output_dir, "*.jpg")))


def infer_from_folder(folder_path, output_json):
    images = sorted(glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png")))
    detectors = setup_detectors()
    all_results = [run_pipeline(img, detectors) for img in images]
    export_batch(all_results, images, output_file=output_json)


def infer_from_gcs(bucket_name, prefix, local_dir, output_json):
    print(f"\nDownloading images from GCS bucket: gs://{bucket_name}/{prefix}")
    os.makedirs(local_dir, exist_ok=True)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    downloaded = []

    for blob in blobs:
        if blob.name.endswith((".jpg", ".jpeg", ".png")):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            downloaded.append(local_path)

    if not downloaded:
        print("No images downloaded.")
        return

    detectors = setup_detectors()
    all_results = [run_pipeline(img, detectors) for img in downloaded]
    export_batch(all_results, downloaded, output_file=output_json)


def infer_from_video_to_label_studio(video_path, output_json, temp_dir=None):
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    try:
        frame_paths = infer_from_video(video_path, output_dir=temp_dir)
        detectors = setup_detectors()
        all_results = [run_pipeline(img, detectors) for img in frame_paths]
        export_batch(all_results, frame_paths, output_file=output_json)
    finally:
        shutil.rmtree(temp_dir)

import argparse
def main():
    parser = argparse.ArgumentParser(description="Run two-step detection pipeline from different sources")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", help="Path to a folder of images")
    group.add_argument("--video", help="Path to a video file")
    group.add_argument("--gcs", nargs=2, metavar=("BUCKET_NAME", "PREFIX"), help="GCS bucket and prefix")

    parser.add_argument("--output", required=True, help="Path to output JSON file for Label Studio")
    parser.add_argument("--temp", default="tmp_frames", help="Temporary folder for extracted frames or downloads")

    args = parser.parse_args()

    if args.folder:
        print(f"\nRunning on local image folder: {args.folder}")
        infer_from_folder(args.folder, args.output)
    elif args.video:
        print(f"\nRunning on video file: {args.video}")
        infer_from_video_to_label_studio(args.video, args.output, temp_dir=args.temp)
    elif args.gcs:
        bucket, prefix = args.gcs
        print(f"\nRunning on GCS bucket: gs://{bucket}/{prefix}")
        infer_from_gcs(bucket, prefix, args.temp, args.output)

if __name__ == "__main__":
    main()
# from inference_runner import infer_from_folder
# infer_from_folder("my_images", "output_folder/label_studio_export.json")
# from inference_runner import infer_from_video_to_label_studio
# infer_from_video_to_label_studio("footage.mp4", "video_results.json")

# from inference_runner import infer_from_gcs
# infer_from_gcs(
#     bucket_name="my-gcs-bucket",
#     prefix="city_images/",
#     local_dir="downloaded_frames",
#     output_json="gcs_results.json"
# )