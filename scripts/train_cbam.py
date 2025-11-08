import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import torch.nn as nn
from models.cbam import CBAM

model = YOLO("yolov5s.pt")

inserted = False
for m in reversed(list(model.model.modules())):
    if isinstance(m, nn.Conv2d) and m.out_channels >= 256:
        ch = m.out_channels
        parent = m
        break

if isinstance(model.model, nn.Sequential):
    ch = 256
    model.model.append(CBAM(ch))
    inserted = True

print(f"CBAM inserted: {inserted}")
model.train(
    data="coco128.yaml",   g
    epochs=25,
    imgsz=640,
    batch=16,
    name="cbam_yolov5s",
)
