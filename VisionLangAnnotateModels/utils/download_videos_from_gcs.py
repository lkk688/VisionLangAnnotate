import os
from google.cloud import storage

def download_all_videos_from_gcs(bucket_name, gcs_prefix="", local_download_dir="downloads", file_extensions=(".mp4", ".avi", ".mov", ".mkv")):
    """
    Download all video files from a GCS bucket, preserving the original folder structure.

    Args:
        bucket_name (str): Name of the GCS bucket.
        gcs_prefix (str): Optional prefix/folder path inside the bucket.
        local_download_dir (str): Local folder where videos will be saved.
        file_extensions (tuple): Video file extensions to download.

    Returns:
        List of downloaded local file paths.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_prefix)

    downloaded_files = []
    total_size = 0

    for blob in blobs:
        if blob.name.lower().endswith(file_extensions):
            # Build local path by appending blob.name to local_download_dir
            local_path = os.path.join(local_download_dir, blob.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Check if file already exists and has the same size
            if os.path.exists(local_path) and os.path.getsize(local_path) == blob.size:
                print(f"⏩ Skipping (already exists): {blob.name}")
                downloaded_files.append(local_path)
                continue
                
            # Download the file
            print(f"⬇️ Downloading: {blob.name} ({format_size(blob.size)})")
            blob.download_to_filename(local_path)
            downloaded_files.append(local_path)
            total_size += blob.size
            print(f"✅ Downloaded: {blob.name} → {local_path}")

    print(f"\nDownload complete! Downloaded {len(downloaded_files)} files ({format_size(total_size)})")
    return downloaded_files

def format_size(size_bytes):
    """Format file size in a human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

def list_bucket_contents(bucket_name, gcs_prefix="", file_extensions=None):
    """
    List all files in a GCS bucket with optional filtering by extension.
    
    Args:
        bucket_name (str): Name of the GCS bucket.
        gcs_prefix (str): Optional prefix/folder path inside the bucket.
        file_extensions (tuple): Optional file extensions to filter by.
        
    Returns:
        List of blob names that match the criteria.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    
    matching_files = []
    folders = set()
    
    for blob in blobs:
        # Track folders
        parts = blob.name.split('/')
        for i in range(len(parts)):
            if i > 0:  # Skip the empty string before the first slash
                folder = '/'.join(parts[:i])
                if folder:
                    folders.add(folder)
        
        # Check if file matches extensions
        if file_extensions is None or blob.name.lower().endswith(file_extensions):
            matching_files.append(blob.name)
    
    return matching_files, sorted(list(folders))

if __name__ == "__main__":
    # Configuration
    BUCKET_NAME = "roadsafetyparkingcompliance"
    GCS_PREFIX = "Parking compliance Vantrue dashcam"
    LOCAL_DOWNLOAD_DIR = "./output/dashcam_videos"
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
    
    # Create output directory
    os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
    
    # Optional: List contents before downloading
    print(f"Listing contents of gs://{BUCKET_NAME}/{GCS_PREFIX}...")
    files, folders = list_bucket_contents(BUCKET_NAME, GCS_PREFIX, VIDEO_EXTENSIONS)
    print(f"Found {len(files)} video files in {len(folders)} folders")
    
    # Download all videos
    print(f"\nDownloading all videos from gs://{BUCKET_NAME}/{GCS_PREFIX} to {LOCAL_DOWNLOAD_DIR}...")
    downloaded = download_all_videos_from_gcs(
        bucket_name=BUCKET_NAME,
        gcs_prefix=GCS_PREFIX,
        local_download_dir=LOCAL_DOWNLOAD_DIR,
        file_extensions=VIDEO_EXTENSIONS
    )