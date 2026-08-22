# Dataset Conversion Report

## Conversion Summary
- **Original image count:** 5708
- **Original annotation count:** 16333
- **Valid polygon count (converted):** 13082
- **Converted bounding-box count:** 16333
- **Excluded annotation count:** 0
- **Images affected by excluded annotations:** 0

## Data Splits
- **Train count:** 5330
- **Valid count:** 201
- **Test count:** 177

## Excluded Annotation Reasons
Check `excluded_annotations.csv` for detailed rows. Common reasons: Insufficient coordinates, non-numeric values, or degenerate boxes.

## Class Distribution Before vs After

| Class Name | Annotations Before | Annotations After |
|---|---|---|
| bolts | 5629 | 5629 |
| crack | 1006 | 1006 |
| flaking | 6017 | 6017 |
| joints | 132 | 132 |
| sheling | 1317 | 1317 |
| spalling | 2232 | 2232 |
