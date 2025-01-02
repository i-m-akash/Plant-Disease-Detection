# Potato Plant Disease Detection Using CNN

This project involves detecting diseases in potato plants (e.g., early blight, late blight, or healthy leaves) using a Convolutional Neural Network (CNN). It leverages image classification techniques to aid farmers in identifying plant diseases at an early stage, minimizing crop losses.

---

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Prediction](#prediction)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Features

- **Image Classification**: Classifies potato plant leaves into categories (e.g., early blight, late blight, healthy).
- **Real-time Detection**: Detects diseases using images of plant leaves.
- **Visualization**: Displays training and validation metrics.
- **Deployment Ready**: Export the trained model for web or mobile integration.

---

## Dataset

- The dataset used is the **PlantVillage Dataset** or any dataset containing labeled images of potato plant diseases.
- Ensure the dataset is organized into `train`, `validation`, and `test` directories with subfolders for each class (e.g., `healthy`, `early_blight`, `late_blight`).

---

## Model Architecture

The CNN architecture consists of:

1. **Convolution Layers**: Extract spatial features from images.
2. **Pooling Layers**: Reduce dimensionality while retaining key features.
3. **Dropout Layers**: Prevent overfitting.
4. **Dense Layers**: Perform classification based on extracted features.

---

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow, Keras, NumPy, Matplotlib, OpenCV

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/potato-disease-detection.git
   cd potato-disease-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model

1. Organize your dataset:
   ```plaintext
   dataset/
       train/
           healthy/
           early_blight/
           late_blight/
       validation/
           healthy/
           early_blight/
           late_blight/
       test/
           healthy/
           early_blight/
           late_blight/
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```

### Predicting with the Model

1. Use the trained model to predict a single image:
   ```bash
   python predict.py --image path_to_image.jpg
   ```

---

## Training and Evaluation

- **Training Metrics**: Accuracy and loss are tracked for each epoch.
- **Validation Metrics**: Model performance is validated after each epoch.
- **Test Evaluation**: Final model accuracy is evaluated on the test dataset.

---

## Prediction

Use the `predict.py` script to test the model on new images:
```bash
python predict.py --image path_to_image.jpg
```

---

## Results

- Achieved an accuracy of **96%** on the validation dataset.
- Confusion matrix and class-wise accuracy available in the evaluation report.

---

## Future Enhancements

- **Expand Dataset**: Add more classes and samples for better generalization.
- **Real-time Integration**: Integrate with IoT devices for field applications.
- **Mobile App Deployment**: Use TensorFlow Lite for Android/iOS applications.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

For contributions or issues, feel free to raise a pull request or open an issue in the repository.

