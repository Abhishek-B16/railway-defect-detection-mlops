import os
import shutil
from roboflow import Roboflow
import getpass

def download_and_filter_v5_dataset():
    print("=== Phase 1: Downloading HD Dataset from Roboflow ===")
    print("Go to https://app.roboflow.com/ and sign in/create a free account.")
    print("Click on your profile -> Settings -> Roboflow API -> Copy your Private API Key.")
    
    api_key = getpass.getpass("Paste your Roboflow API Key here (it will be hidden): ")
    
    try:
        rf = Roboflow(api_key=api_key.strip())
        project = rf.workspace("railway-defect-detection-t5z2a").project("railway-track-fault-detection-z4pze")
        version = project.version(1)
        dataset = version.download("yolov8")
        print(f"Dataset downloaded to: {dataset.location}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # TODO: Add filtering logic here once dataset is downloaded
    print("\nDataset downloaded successfully! We can now proceed to filtering out bad/noisy images.")

if __name__ == "__main__":
    download_and_filter_v5_dataset()
