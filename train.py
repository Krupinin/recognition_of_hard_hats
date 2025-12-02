import yaml
import os
from ultralytics import YOLO

def main():
    # Check if ultralytics is installed
    try:
        import ultralytics
    except ImportError:
        print("Ultralytics not installed. Please install it with: pip install ultralytics")
        return

    # Create data.yaml
    config = {
        "train": "Dataset/images/train",
        "val": "Dataset/images/val",
        "test": "Dataset/images/test",
        "nc": 3,
        "names": ['helmet', 'head', 'person']
    }
    with open("data.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print("data.yaml created.")

    # Train model
    model = YOLO('yolov8m.pt')  # load a pretrained model
    model.train(data='data.yaml', epochs=20, lr0=0.01, batch=-1, workers=4)  # train the model

    print("Training completed.")

if __name__ == "__main__":
    main()