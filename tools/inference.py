import onnxruntime as ort
import numpy as np
import cv2
import argparse

def preprocess(image_path, input_shape=(640, 640)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize and Pad logic (Simplified for demo: Letterbox is better, but this is bare minimum)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_shape)
    
    # HWC to CHW, Normalize to 0-1
    img_in = img_resized.transpose((2, 0, 1)) / 255.0
    img_in = np.expand_dims(img_in, axis=0).astype(np.float32)
    
    return img_in, img

def postprocess(output, original_img):
    # YOLOv8 output shape is roughly (1, 84, 8400) -> (Batch, 4+Classes, Proposals)
    # This is a raw output that needs Non-Max Suppression (NMS)
    # For this demo, we'll just print the raw shape to prove inference ran.
    # Writing a full NMS from scratch in numpy is lengthy; usually we rely on library helpers.
    print(f"Raw Output Shape: {output[0].shape}")
    print("Inference Successful! (Raw tensor obtained)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    # 1. Load Session
    print(f"Loading session for {args.model}...")
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    
    # 2. Preprocess
    img_input, original_img = preprocess(args.image)
    
    # 3. Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    
    # 4. Postprocess
    postprocess(outputs, original_img)

if __name__ == "__main__":
    main()
