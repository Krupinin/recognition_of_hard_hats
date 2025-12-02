from ultralytics import YOLO
import os

def main():
    # Determine model path: use best.pt if trained, else yolov8n.pt
    model_path = 'runs/detect/train/weights/best.pt'
    if not os.path.exists(model_path):
        model_path = 'yolov8n.pt'
        print(f"Using default model: {model_path}")
    else:
        print(f"Using trained model: {model_path}")

    # Load model
    model = YOLO(model_path)

    # Get all images in Dataset/images
    images_dir = 'Dataset/images'
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Test on each image and print filename
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        # print(f"Testing: {image_file}")

        # Predict on the image
        results = model.predict(source=image_path, conf=0.25, save=False, verbose=False)

        # Optionally, you can add more logging here, like number of detections
        # For example: print(f"  Detections: {len(results[0].boxes)}")
        # But sticking to printing names as requested

if __name__ == "__main__":
    main()
