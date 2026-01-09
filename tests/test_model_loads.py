from ultralytics import YOLO

def test_model_loads():
    # Use any small official model to test loading works
    model = YOLO("yolov5su.pt")
    assert model is not None
