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

    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print("Loaded configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Train model
    model = YOLO(config['model'])  # load the selected model

    train_kwargs = {
        'data': 'data.yaml',
        'epochs': config['epochs'],
        'lr0': config['lr0'],
        'batch': config['batch'],
        'workers': config['workers'],
        'imgsz': config['imgsz']
    }

    if config['half']:
        train_kwargs['half'] = True

    if config['device']:
        train_kwargs['device'] = config['device']

    if config['patience'] > 0:
        train_kwargs['patience'] = config['patience']

    model.train(**train_kwargs)  # train the model

    print("Training completed.")

if __name__ == "__main__":
    main()
