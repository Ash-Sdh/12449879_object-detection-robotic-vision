import numpy as np
from ultralytics import YOLO

def test_inference_runs():
    model = YOLO("yolov5su.pt")
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(img, verbose=False)
    assert results is not None
    assert len(results) > 0
