# Fashion MNIST Classifier Frontend

This repository contains a **Streamlit-based frontend application** for the Fashion MNIST dataset. It enables users to upload images of fashion items, preprocess them, and classify them using a pre-trained deep learning model. 

## Features

- **User-friendly Interface**:
  - Upload images of fashion items directly from your device.
  - View the uploaded image and its processed version (grayscale and resized to `28x28` pixels).
  
- **Image Preprocessing**:
  - Converts images to grayscale.
  - Resizes images directly to `28x28` pixels for compatibility with the Fashion MNIST model.
  - Normalizes pixel values for better prediction accuracy.

- **Model Integration**:
  - Predicts the class of the uploaded fashion item using a TensorFlow-based pre-trained model.
  - Outputs the predicted class along with the confidence score.

- **Dynamic Visualization**:
  - Displays a bar chart of predicted confidence scores for all Fashion MNIST classes.

- **Custom Background**:
  - A visually appealing background is set using a local image file.

## Classes in Fashion MNIST

The application classifies images into the following categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## How It Works

1. Upload an image of a fashion item in one of the supported formats (`png`, `jpg`, `jpeg`, `webp`).
2. The app preprocesses the image:
   - Converts it to grayscale.
   - Resizes it to `28x28` pixels.
   - Normalizes the pixel values.
3. The preprocessed image is passed to a TensorFlow model, which predicts the class of the item.
4. The app displays the predicted class, confidence score, and a confidence bar chart.

## Getting Started

### Prerequisites
- Python 3.8 or later
- TensorFlow
- Streamlit
- PIL (Pillow)
- pandas
- numpy

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name/fashion-mnist-classifier.git
   cd fashion-mnist-classifier
   
2. Install dependencies, Place pre-trained TensorFlow model file (modelAarib.h5) in the root directory.

3. Start the Streamlit app by running:
   ```bash
   streamlit run app.py
