import time, torch
from ultralytics import YOLO
WEIGHTS = "runs/detect/baseline_yolov5s/weights/best.pt"
model = YOLO(WEIGHTS)
img = torch.zeros(1, 3, 640, 640)  
for _ in range(10):
    model.predict(img, verbose=False)

N = 50
t0 = time.perf_counter()
for _ in range(N):
    model.predict(img, verbose=False)
t1 = time.perf_counter()

lat_ms = (t1 - t0) * 1000 / N
fps = 1000.0 / lat_ms
print(f"Latency: {lat_ms:.2f} ms/img  |  FPS: {fps:.1f}")
