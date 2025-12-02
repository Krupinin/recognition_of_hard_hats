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

    # Test on each image and print filename if person without hard hat is detected
    images_with_person_without_hardhat = []

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        print(f"Testing: {image_file}")

        # Predict on the image
        results = model.predict(source=image_path, conf=0.25, save=False, verbose=False)

        # Get classes of detected objects (0: helmet, 1: head, 2: person)
        detected_classes = [int(box.cls[0]) for box in results[0].boxes] if results[0].boxes else []

        # Check if there are head or person detections and no helmet
        has_person = any(cls in [1, 2] for cls in detected_classes)
        has_hardhat = any(cls == 0 for cls in detected_classes)

        if has_person and not has_hardhat:
            images_with_person_without_hardhat.append(image_file)
            print(f"  Person without hardhat detected in {image_file}")
        else:
            print(f"  Detections: {len(results[0].boxes)}")

    # Output the names of images with person without hard hat
    if images_with_person_without_hardhat:
        print("\nImages where a person without a hard hat was detected:")
        for img in images_with_person_without_hardhat:
            print(f"  {img}")
    else:
        print("\nNo images found with person without hard hat.")

if __name__ == "__main__":
    main()
