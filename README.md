# CNN_MNIST
# Handwritten Digit Recognition using CNN (MNIST Dataset)

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0-9) using the MNIST dataset. The CNN model is trained on grayscale images of size 28x28 pixels and can predict digits from new input images.

---

## Features

- Larger CNN architecture for improved accuracy.
- Training on MNIST dataset for 10 epochs.
- Saves the trained model to Google Drive.
- Loads the saved model to predict digits from custom images.
- Image preprocessing includes resizing, normalization, and color inversion.
- Visualizes input images before prediction.

---

## Installation

Install necessary libraries using:

```bash
!pip install opencv-python-headless
```
Make sure TensorFlow, Keras, NumPy, OpenCV, and Matplotlib are installed. TensorFlow usually includes Keras.

---

## Usage

### Training the Model

- Load the MNIST dataset.
- Normalize and reshape data for CNN input.
- Define and compile a CNN with convolutional, pooling, dropout, flatten, and dense layers.
- Train for 10 epochs with a batch size of 200.
- Evaluate the model on test data.
- Save the trained model to Google Drive as `digit_model.h5`.

### Predicting New Digits

- Mount Google Drive in your Colab environment.
- Provide the path to a grayscale image of a handwritten digit.
- Preprocess the image: resize to 28x28, invert colors, normalize pixel values, and reshape.
- Load the saved CNN model.
- Predict the digit class.
- Display the input image and predicted digit.

---

## File Structure

- `digit_model.h5`: Saved trained model.
- `number.jpeg`: Sample input image for prediction.
- Python script containing training and prediction code.

---

## Model Architecture

- Conv2D layer with 30 filters, 5x5 kernel, ReLU activation.
- MaxPooling2D with 2x2 pool size.
- Conv2D layer with 15 filters, 3x3 kernel, ReLU activation.
- MaxPooling2D with 2x2 pool size.
- Dropout layer with 20% rate.
- Flatten layer.
- Dense layer with 128 neurons, ReLU activation.
- Dense layer with 50 neurons, ReLU activation.
- Output Dense layer with 10 neurons (one per digit class), softmax activation.

---

## Notes

- Ensure Google Drive is mounted correctly in Colab before saving/loading models.
- Input images should be clear, handwritten digits on a white background.
- Color inversion (`cv2.bitwise_not`) is important to match MNIST digit style.



