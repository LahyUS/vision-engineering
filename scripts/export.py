from ultralytics import YOLO
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to input .pt model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    try:
        model = YOLO(args.model)
        
        print("Exporting to ONNX...")
        # export format='onnx' automatically handles simplification and opset
        path = model.export(format="onnx", opset=args.opset)
        
        print(f"Export Success! Model saved to: {path}")
        
    except Exception as e:
        print(f"Error during export: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
