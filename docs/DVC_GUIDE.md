# DVC Guide: Dataset Versioning for Railway Defect Detection

This project uses Data Version Control (DVC) to track and version our machine learning datasets, alongside Git for our code.

## Why DVC is Being Used
Machine learning datasets (especially computer vision datasets with thousands of high-resolution images) are typically too large to be committed directly into Git. Git is optimized for tracking text-based source code, and committing GBs of images causes severe performance degradation and bloats the repository.

DVC solves this by acting like Git for data. It tracks the exact state, hashes, and structure of your dataset without storing the actual files in Git. Instead, the data itself is offloaded to a designated storage location (local disk cache or remote cloud buckets like S3, Google Drive, Azure Blob), while Git only tracks lightweight DVC metadata files (`.dvc`).

## What Dataset Versioning Means
Dataset versioning guarantees **reproducibility**. Just as Git allows you to revert your code to a state from a month ago, DVC allows you to revert your data to its exact state at that same time.
- If we add 1,000 new images to `dataset_detection/` tomorrow and retrain the model, we want to know exactly what data produced the new model.
- Versioning links a specific dataset state (via a DVC tracking file) to a specific Git commit.

## How the Current Dataset is Versioned
1. The raw bounding box dataset is stored in the `dataset_detection/` folder.
2. We ran `dvc add dataset_detection`, which did two things:
   - Added `dataset_detection/` to `.gitignore` so Git ignores the heavy images and labels.
   - Created a metadata file named `dataset_detection.dvc` containing the MD5 hash of the entire folder.
3. We commit `dataset_detection.dvc` and the `.gitignore` into Git.
4. Git tracks the `.dvc` file. DVC tracks the actual data files internally.

## How to Reproduce the Dataset
If a new developer clones this Git repository, they will NOT see the `dataset_detection/` folder immediately (since it's ignored by Git).
Instead, they will only see `dataset_detection.dvc`.

To retrieve the actual data, the developer will:
1. `git clone <repo_url>`
2. `dvc pull`

DVC will read `dataset_detection.dvc`, connect to the configured DVC remote storage (or local cache), download the exact image and label files that match the hash, and place them in the `dataset_detection/` folder, reproducing the dataset exactly as it was when the commit was made.

## How a Future Dataset Update Creates a New Version
When more defect images are collected or labels are refined in the future:
1. Update the files inside `dataset_detection/`.
2. Run `dvc add dataset_detection`. DVC calculates a new MD5 hash and updates the `dataset_detection.dvc` file.
3. Commit the updated `dataset_detection.dvc` file to Git:
   ```bash
   git add dataset_detection.dvc
   git commit -m "Update dataset with 500 new crack images"
   ```
4. Run `dvc push` to upload the new files to the remote DVC storage.

The new dataset version is now securely linked to the new Git commit.
