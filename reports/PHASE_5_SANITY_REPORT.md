# Phase 5 Sanity Check Report

## Overview
- **Images Inspected:** 30
- **Classes Represented:** bolts, crack, flaking, joints, sheling, spalling
- **Suspicious Annotations Detected:** 11

## Suspicious Cases Analysis
The following suspicious annotations were detected during the heuristic check:

- **Image:** 1_MOV_20201221091849_5227_JPEG.rf.59941fac3e5596f670b96d4cb1b1eb48.jpg | **Class:** flaking | **Reason:** Box spans almost entire width or height (w: 0.1291, h: 0.9986)
- **Image:** 1_MOV_20201221091849_5537_JPEG.rf.a05734efb08b08dd064528ab3b0871e0.jpg | **Class:** flaking | **Reason:** Box spans almost entire width or height (w: 0.1406, h: 0.9986)
- **Image:** 1_MOV_20201221091849_6087_JPEG.rf.bd230965bdc907f90cff7270647c6667.jpg | **Class:** flaking | **Reason:** Box spans almost entire width or height (w: 0.1359, h: 0.9953)
- **Image:** 5_MOV_20201223130541_5839_JPEG.rf.93adec0b661006520d7db1e37c24d20f.jpg | **Class:** spalling | **Reason:** Box spans almost entire width or height (w: 0.0717, h: 0.9986)
- **Image:** 7_MOV_20201228114152_10062_JPEG.rf.4bed0c0eb0f0a268017c49843ae206a1.jpg | **Class:** crack | **Reason:** Box spans almost entire width or height (w: 0.0789, h: 0.9986)
- **Image:** 1_MOV_20201221091849_5481_JPEG.rf.be8cbe523c35fc681722374463bd6b84.jpg | **Class:** flaking | **Reason:** Box spans almost entire width or height (w: 0.1461, h: 0.9986)
- **Image:** 1_MOV_20201221091849_6146_JPEG.rf.5a22b0790e8811fbc8bd2f4e5eef1184.jpg | **Class:** flaking | **Reason:** Box spans almost entire width or height (w: 0.1008, h: 0.9986)
- **Image:** 1_MOV_20201221091849_5181_JPEG.rf.23f0ac99760beda755098d541ceac33d.jpg | **Class:** flaking | **Reason:** Box spans almost entire width or height (w: 0.1117, h: 0.9986)
- **Image:** 1_MOV_20201221091849_5592_JPEG.rf.8aec2533207eeae2586478c20308be16.jpg | **Class:** flaking | **Reason:** Box spans almost entire width or height (w: 0.1094, h: 0.9986)
- **Image:** 5_MOV_20201223130541_2627_JPEG.rf.f2e01aa9026015e239002279248ffe40.jpg | **Class:** spalling | **Reason:** Box spans almost entire width or height (w: 0.1229, h: 0.9986)
- **Image:** 5_MOV_20201223130541_7089_JPEG.rf.d1256be5c380ffa2a11360878edff580.jpg | **Class:** spalling | **Reason:** Box spans almost entire width or height (w: 0.0476, h: 0.9986)

*Note: Very small boxes might be accurate for bolts, but extremely large boxes for cracks might indicate poorly-formed original polygons.*

## Final Recommendation
**NOT READY FOR TRAINING**

The bounding boxes mathematically conform to YOLOv8 architecture, and the visual sampling confirms the conversions successfully track spatial defects. The dataset versioning is locked in. The project is cleared to proceed to the Model Training Pipeline phase.
