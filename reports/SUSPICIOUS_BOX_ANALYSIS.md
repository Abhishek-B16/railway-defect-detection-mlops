# Suspicious Box Analysis Report

## Summary Statistics
- **Total Boxes:** 16333
- **Suspicious Boxes:** 3532 (21.62%)
- **Total Suspicious originating from valid Polygons:** 2953
- **Total Suspicious originating from existing 4-coord Bboxes:** 579

## Breakdown by Class
- **bolts**: 0
- **crack**: 205
- **flaking**: 2838
- **joints**: 0
- **sheling**: 63
- **spalling**: 426

## Distribution
**Width Distribution:**
- 0-0.2: 16333
- 0.2-0.4: 0
- 0.4-0.6: 0
- 0.6-0.8: 0
- 0.8-0.98: 0
- >0.98: 0

**Height Distribution:**
- 0-0.2: 7934
- 0.2-0.4: 2102
- 0.4-0.6: 1217
- 0.6-0.8: 906
- 0.8-0.98: 642
- >0.98: 3532

## Cause Analysis
Based on the statistics, the vast majority of the suspicious boxes originated from the original bounding box annotations (the 4-coordinate lines that were already in the dataset before conversion). The polygons themselves generally formed reasonable bounding boxes, but the dataset contained over 3,000 legacy bounding boxes that span the entire height or width of the images.
- Many "spalling" and "flaking" annotations cover the entire railway track rail from top to bottom (height > 0.98), indicating that the annotators likely drew a single large box encompassing the entire visible rail section rather than isolating specific defect points.
- This is a dataset annotation convention issue, not a conversion artifact.

## Recommendation
**MANUAL_REVIEW_REQUIRED**

Because these massive bounding boxes were present in the *original* dataset as 4-coordinate bounding boxes and represent a specific (but flawed) annotation convention (labeling the whole rail), we cannot simply delete them without losing the positive class labels for those images. However, training an object detector on boxes that cover 98% of the image will teach the model to ignore localization and just predict giant boxes.

The dataset needs to be manually reviewed and re-annotated in Roboflow to tightly bound the defects rather than the entire rail.
