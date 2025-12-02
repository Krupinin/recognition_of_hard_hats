# Hard Hat Detection Project

This project implements a hard hat detection system using YOLOv8, based on the code from the Jupyter notebook.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:**
   ```bash
   python data_prep.py
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Run detection on an image with notifications:**
   ```bash
   python detect.py path/to/your/image.png
   ```

   The script will print warnings if persons without helmets are detected.

## Files

- `data_prep.py`: Prepares dataset by converting annotations and splitting into train/val/test.
- `train.py`: Trains the YOLOv8 model.
- `detect.py`: Runs inference on an image and checks for persons without helmets.
- `requirements.txt`: List of Python packages required.

## Detection Logic

The notification checks if detected 'person' or 'head' objects have overlapping 'helmet' detections (IoU > 0.5). If not, a warning is printed.
