import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


# Load the saved model
@st.cache_resource
def load_model():
    """
    Load the trained model for predicting Fashion MNIST classes.
    """
    return tf.keras.models.load_model("modelAarib.h5")  # Replace with the augmented model's filename if necessary


model = load_model()


# Function to preprocess the input image
def preprocess_image(image):
    """
    Preprocess the input image to match the model's expected input:
    - Convert to grayscale.
    - Resize to 28x28 without intermediate downscaling.
    - Normalize pixel values.
    """
    # Convert to grayscale
    image = image.convert("L")

    # Resize directly to 28x28
    image = image.resize((28, 28), Image.NEAREST)

    # Normalize pixel values
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Ensure the input shape is (1, 28, 28, 1)
    image_array = np.expand_dims(image_array, axis=(0, -1))
    return image_array


# Class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Add a custom background using local CSS
def add_background(image_file):
    """
    Adds a background image to the Streamlit app.
    """
    background_css = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{image_file});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)


# Convert local image to Base64 for embedding
import base64


def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Streamlit app design
background_image_path = "249.jpg"  # Replace with your local image file path
add_background(get_base64(background_image_path))

st.title("Fashion MNIST Classifier")
st.write("Upload an image of a fashion item, and the model will classify it.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Pixelate and resize directly to 28x28
    pixelated_image = image.convert("L").resize((28, 28), Image.NEAREST)
    st.image(pixelated_image, caption="Pixelated and Resized Image (28x28)", use_container_width=True)

    st.write("Processing the image...")

    # Preprocess and predict
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)

    # Get predicted class and confidence
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100

    # Display the prediction results
    st.write(f"### Predicted Class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Create a DataFrame for bar chart with class names
    prediction_data = pd.DataFrame({
        "Class": class_names,
        "Confidence": predictions[0]
    })

    # Display the bar chart with class names
    st.bar_chart(prediction_data.set_index("Class"))
