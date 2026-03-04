# 🔢 MNIST Handwritten Digit Classifier

Classify handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset — and even predict digits from your own custom images!

---

## 🧠 About the Project

This beginner-friendly deep learning project demonstrates how to build and train a CNN to recognize handwritten digits using the MNIST dataset. The model achieves **98%+ accuracy** and even allows predictions on your own uploaded digit images!

Perfect for those learning CNNs, image classification, and model deployment in TensorFlow/Keras.

---

## 🚀 Features

- 📦 Load and normalize MNIST dataset  
- 🧠 Build a CNN from scratch with TensorFlow/Keras  
- 📊 Evaluate with accuracy and test predictions  
- 🖼️ Predict digits from custom grayscale images  
- 🧪 Visualize digits and model predictions  

---

## 🛠️ Tech Stack

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Pillow (PIL)  
- Matplotlib  

---

## 📁 Project Structure

```
mnist-digit-classifier/
├── digit_classifier.py       # Main script
├── digit_classifier.ipynb    # Jupyter Notebook version
├── README.md                 # Project overview and instructions
├── requirements.txt          # Required libraries
└── images/
    ├── sample_digit0.png     # Example test image
    └── sample_digit7.png     # Example test image
```

---

## 💻 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/saadtoorx/mnist-digit-classifier.git
cd mnist-digit-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Classifier

```bash
python digit_classifier.py
```

### 4. Predict Custom Image

```text
Enter path to the image with extension: images/sample_digit7.png
🧠 Predicted Digit: 7
