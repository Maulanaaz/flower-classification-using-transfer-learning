# 📓 Experimentation Notebooks

This directory contains the Jupyter Notebooks used for the development, training, and evaluation of the Flower Classification model.

## 📄 File Description

| File Name | Description |
| :--- | :--- |
| **`flower_classification_mobilenetv2.ipynb`** | The main notebook containing the end-to-end pipeline: Data Loading, Augmentation, MobileNetV2 Transfer Learning, and TFLite Conversion. |

## 🧪 Notebook Workflow

The notebook follows a structured machine learning pipeline:

1.  **Data Loading & Preprocessing**
    * Loading images from the dataset.
    * Applying `ImageDataGenerator` for augmentation (Rotation, Zoom, Flip).
    * Resizing images to `224x224`.
2.  **Model Construction**
    * Importing pre-trained **MobileNetV2** (ImageNet weights).
    * Adding custom classification head (Dense + Dropout).
3.  **Training Phase**
    * Compiling with `Adam` optimizer.
    * Using Callbacks:  `ModelCheckpoint`,  `EarlyStopping` (patience=5)`, and `ReduceLR`.
4.  **Evaluation**
    * Visualizing Accuracy & Loss curves.
5.  **Deployment Preparation**
    * Converting the trained Keras model (`.h5`) to TensorFlow Lite (`.tflite`).
