import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import torch.nn as nn
from models.cbam import CBAM

# Load pretrained YOLOv5s
model = YOLO("yolov5s.pt")

# Find last conv layer before Detect and wrap with CBAM (simple, robust)
# NOTE: Ultralytics model modules are nested; we hook near the end:
# We'll scan for the largest feature extractor block and append CBAM.
# If this fails on a future version, fallback: skip CBAM and just train baseline.

inserted = False
for m in reversed(list(model.model.modules())):
    if isinstance(m, nn.Conv2d) and m.out_channels >= 256:
        ch = m.out_channels
        parent = m
        # Wrap via a Sequential only if parent has sequential context
        # Safer: add a CBAM block at model.model[-2] if available
        break

# Practical approach: append a small CBAM at the end of the backbone via a nn.Sequential
# (Ultralytics model.model is a nn.Sequential)
if isinstance(model.model, nn.Sequential):
    ch = 256
    model.model.append(CBAM(ch))
    inserted = True

print(f"CBAM inserted: {inserted}")

# Train config
model.train(
    data="./data/coco_subset.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    name="cbam_yolov5s"
)
