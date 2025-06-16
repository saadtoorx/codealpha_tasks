# 🧠 MNIST Handwritten Digit Classifier

This project is a simple yet powerful Convolutional Neural Network (CNN) model trained on the MNIST dataset to recognize handwritten digits (0-9). It also includes a feature to predict digits from your own image files using the trained model.

---

## 📁 Dataset

- **MNIST Dataset** from `tensorflow.keras.datasets`
- Contains **60,000 training images** and **10,000 test images**
- Each image is **28x28 pixels**, grayscale

---

## 🧠 Model Architecture

- **Input Layer**: (28, 28, 1)
- **Conv2D Layer 1**: 32 filters, (3x3), ReLU
- **MaxPooling Layer**: (2x2)
- **Conv2D Layer 2**: 64 filters, (3x3), ReLU
- **Flatten**
- **Dense Layer**: 64 neurons, ReLU
- **Output Layer**: 10 neurons, Softmax (for digits 0-9)

---

## 🛠️ Tech Stack & Libraries

- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Matplotlib

---

## 🚀 How to Use

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/saadtoorx/MNIST-Digit-Classifier.git
cd MNIST-Digit-Classifier
