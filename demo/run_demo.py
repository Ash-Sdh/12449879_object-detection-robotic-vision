from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Object Detection Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5s.pt",
        help="Path to YOLO model (.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image / video file or webcam index (0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_outputs",
        help="Folder to save results"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading model:", args.model)
    model = YOLO(args.model)

    print("Running inference on:", args.source)
    model.predict(
        source=args.source,
        save=True,
        project=args.output,
        name="results",
        conf=0.25
    )

    print("Demo finished.")
    print(f"Results saved to: {args.output}/results")

if __name__ == "__main__":
    main()
