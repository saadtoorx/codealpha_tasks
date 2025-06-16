import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
#Load MNIST Dataset:
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#Preprocessing or Normalizing the pixel values to be between 0 and 1

train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images into 3 dimensions as cnn excepts each images into such and mnists data is greyscale:
train_images = train_images.reshape((train_images.shape[0], 28,28,1))
test_images = test_images.reshape((test_images.shape[0], 28,28,1))

#Covert the labels into one-hot encoded format:
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels) 
#Building CNN Model:
model = models.Sequential()

#First convolutional layer:
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

#Second convolutional layer:
model.add(layers.MaxPooling2D((2,2)))

#Third convolutional layer:
model.add(layers.Conv2D(64, (3,3), activation='relu'))

#Flatten the 3d output to 1d and add a dense layer:
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#Output layers with 10 neurons (for 10 digit classes)
model.add(layers.Dense(10, activation='softmax'))
#Compile the model:
model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )
#Train the model:
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))
#Make Predictions:
# predictions = model.predict(test_images)

# print(f"Prediction for the first test image: {np.argmax(predictions[0])}")
#Visualization:

# plt.imshow(test_images[1].reshape(28,28), cmap='gray')
# plt.title(f"Predicted label: {predictions[1].argmax()}")
# Function to preprocess the image using PIL
def preprocess_image(image_path):
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Resize to 28x28 pixels
    img = img.resize((28, 28))

    # Invert colors if background is white (optional)
    img = ImageOps.invert(img)

    # Convert to NumPy array
    img_array = np.array(img)

    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0

    # Reshape to (1, 28, 28, 1) as expected by the model
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# Function to predict the digit
def predict_digit(image_path):
    img_input = preprocess_image(image_path)
    prediction = model.predict(img_input)
    predicted_digit = np.argmax(prediction)
    print(f"🧠 Predicted Digit: {predicted_digit}")
    return predicted_digit
# User input
image_path = input("Enter path to the image with extension:")  # Replace with your actual image file
predict_digit(image_path) 