
# Product Detection and Counting on Shelves Using YOLO

## Project Overview

This project aims to develop a YOLO model to accurately detect and count product facings on shelves or in refrigerators. The model is fine-tuned to recognize various products and provide an accurate count.

## Setup and Installation

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv yolov8_env
   source yolov8_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Model Setup

1. Download the pre-trained YOLOv8 weights from [YOLOv8 Weights](https://github.com/ultralytics/yolov5/releases).
2. Place the weights in the appropriate directory:
   ```bash
   mkdir weights
   mv <downloaded_weights> weights/
   ```

## Usage

### Training the Model

1. Prepare your dataset following the YOLO format and update the `data.yaml` configuration file accordingly.

2. To train the model run the training script form command line with the desired arguments. For example:
    ```bash
    python train_yolo.py --model-path <model path> --data-path <data.yaml path>
    ```


### Evaluating the Model

1. Evaluate the model:
   ```python
   # Load the model.
   model = YOLO('runs/detect/yolov8m/weights/best.pt')

   # Evaluate the model.
   results = model.val(
       data='pothole_v8.yaml',
       imgsz=1280,
       name='yolov8s_eval'
   )
   ```

## Results Interpretation

### Output Files

- **Bounding Box Annotations**: The model outputs images with bounding boxes drawn around detected products.
- **Metrics**: Precision, Recall, F1-Score, and other relevant metrics are provided to assess the model's performance.

### Interpreting Metrics

- **Precision**: The percentage of correctly identified products out of all identified products.
- **Recall**: The percentage of correctly identified products out of all actual products.
- **F1-Score**: The harmonic mean of Precision and Recall, providing a balance between the two.

### Visual Output

- Detected products are marked with bounding boxes on the images.
- The count of detected products is displayed on the image.

## Results Interpretation

### Accuracy

To evaluate the accuracy of the model, consider the following metrics:

- **Precision**: Indicates the percentage of correctly detected products out of all detections made by the model.
- **Recall**: Indicates the percentage of actual products detected by the model out of all actual products present in the images.
- **F1-Score**: The harmonic mean of Precision and Recall, providing a single measure of model accuracy.

### Speed

Measure how fast the model processes images to detect and count products. This is crucial for real-time applications.

### Robustness

Evaluate the model’s performance across various lighting conditions and different product arrangements to ensure consistent results.

## Exporting the Model

1. Export for NVIDIA devices:
   ```python
   model.export(format='engine', device='cuda')
   ```

2. Export for edge devices:
   ```python
   model.export(format='onnx', device='cpu')
   ```

## Deliverables

1. **Model Code and Weights**: The complete code for the fine-tuned YOLO model, along with the trained weights.
2. **Scripts and Notebooks**: Any scripts or Jupyter notebooks used for data preprocessing, model training, and evaluation.
3. **Evaluation Report**: A concise report detailing the model’s performance, including various evaluation metrics.
4. **Presentation**: A slide deck summarizing the approach, methodology, results, and key findings.

---

### Placeholders

- `<repository_url>`: Replace with the URL of your project's repository.
- `<repository_name>`: Replace with the name of your project's repository.
- `<downloaded_weights>`: Replace with the path to the downloaded YOLOv8 weights.

Add more specific details in the placeholders where necessary, such as links to the dataset, precise evaluation metrics, and sample results.

Feel free to modify and expand this README file as your project progresses and more details become available.
