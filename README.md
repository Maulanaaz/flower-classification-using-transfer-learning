# 🌺 Flower Classification using MobileNetV2 (Transfer Learning)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green)
![Status](https://img.shields.io/badge/Status-Deployment%20Ready-success)

> **"A high-performance image classification system capable of identifying 14 different flower species using Transfer Learning with MobileNetV2 architecture, optimized for mobile deployment (TF-Lite)."**

## 📌 Project Overview

Building an accurate image classifier from scratch requires massive datasets and computing power. This project solves that challenge by leveraging **Transfer Learning**.

I utilized **MobileNetV2**, a pre-trained Convolutional Neural Network (CNN) originally trained on the ImageNet dataset. By fine-tuning this lightweight architecture, the model achieves high accuracy on specific flower datasets while remaining efficient enough for edge devices (mobile/IoT).

### 🎯 Key Features
* **Transfer Learning Strategy:** Used MobileNetV2 as the feature extractor (base model) with frozen layers, adding a custom dense head for classification.
* **Data Augmentation:** Implemented `ImageDataGenerator` (Rotation, Zoom, Horizontal Flip, Shear) to prevent overfitting and improve model generalization.
* **Callback Mechanisms:** Integrated `EarlyStopping` and `ModelCheckpoint` to ensure the best weights are saved and training stops when convergence is reached.
* **Deployment Ready:** Converted the final trained model into **TensorFlow Lite (.tflite)** format for mobile application integration.

---

## 📂 Dataset Info

* **Classes:** 14 Types of Flowers (e.g., Rose, Sunflower, Tulip, etc.)
* **Preprocessing:** Images resized to `224x224` pixels to match MobileNetV2 input requirements.
* **Tech Stack:** TensorFlow, Keras, NumPy, Matplotlib.

---

## 🧠 Model Architecture

The architecture consists of two main parts:

1.  **Base Model (MobileNetV2):**
    * Pre-trained on ImageNet.
    * Layers are frozen (non-trainable) to retain learned features like edges and textures.
2.  **Classification Head (Custom Layers):**
    * **GlobalAveragePooling2D:** Reduces spatial dimensions.
    * **Dense Layer:** 128 units with ReLU activation.
    * **Dropout (0.5):** To reduce overfitting.
    * **Output Layer:** Softmax activation with 14 units (probabilities).

---

## 📊 Performance Results

The model demonstrated robust performance on unseen data:

| Metric | Training Score | Validation Score |
| :--- | :--- | :--- |
| **Accuracy** | **90.02%** | **93.88%** |
| **Loss** | **0.31** | **0.22** |

> (Note: The validation accuracy surpassing training accuracy suggests that the heavy Data Augmentation and Dropout layers effectively regularized the training process.)*
