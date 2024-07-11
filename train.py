import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLO model.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data configuration file.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training.')
    parser.add_argument('--batch', type=int, default=6, help='Batch size for training.')
    parser.add_argument('--name', type=str, default='yolov8n', help='Name of the training run.')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers for data loading.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = YOLO(args.model_path)

    model.train(
        data=args.data_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        workers=args.workers
    )

if __name__ == "__main__":
    main()