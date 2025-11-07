yolo detect train \
  model=yolov5s.pt \
  data=data/coco_subset.yaml \
  imgsz=640 \
  epochs=25 \
  batch=16 \
  name=baseline_yolov5s
