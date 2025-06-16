🔢 Classify handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset — and even predict digits from your own custom images!

---

🧠 **About the Project**  
This beginner-friendly deep learning project walks you through building and training a CNN to recognize handwritten digits using the popular MNIST dataset. With over 98% accuracy, the model can also classify digits from your own image input — just upload a photo of a digit, and the model does the rest!

---

🚀 **Features**  
📦 Load and preprocess the MNIST dataset  
🧠 Build a CNN from scratch with TensorFlow/Keras  
📊 Train and evaluate model accuracy  
🖼️ Predict digits from user-uploaded images  
🎨 Visualize digit images and predictions  

---

🛠️ **Tech Stack**  
- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Pillow (PIL)  
- Matplotlib  

---

📁 **Project Structure**  
```
mnist-digit-classifier/
├── digit_classifier.py       # Main
├── digit_classifier.ipynb    # Jupyter Notebook
├── README.md                 # Project overview and guide
├── requirements.txt          # Required Python libraries
├── images                    # images folder
  └── sample_digit0.png       # Example input image
  └── sample_digit7.png       # Example input image
```

---

💻 **How to Run**

**1. Clone the Repository**
```bash
git clone https://github.com/saadtoorx/mnist-digit-classifier.git
cd mnist-digit-classifier
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Classifier**
```bash
python main.py
```

**4. Provide an Image Path When Prompted**
```
Enter path to the image with extension: my_digit.png
🧠 Predicted Digit: 7
```

> 💡 Tip: Use a 28x28 pixel grayscale image with the digit written in black on a white background (or vice versa — the model auto-inverts it if needed).

---

📷 **Sample Output**  
- Accuracy: ~98% on test data  
- Predicts digit from user-uploaded image  
- Model output:
  ```
  🧠 Predicted Digit: 4
  ```

---

🧾 **License**  
This project is licensed under the MIT License.

---

👤 **Author**  
Made with ❤️ by [@saadtoorx](https://github.com/saadtoorx)

If you found this useful, feel free to fork, explore, and ⭐ the repo!
