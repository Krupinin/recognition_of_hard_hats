# Hard Hat Detection Project

This project implements a hard hat detection system using YOLOv8. Using the web service, you can upload an array of photographs and receive a report on violations.

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

4. **Start web service**
   ```bash
   cd web_interface
   pip install -r requirements-web.txt
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Data source

[Link to kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection/data)

