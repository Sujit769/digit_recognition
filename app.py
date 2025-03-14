import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from streamlit_drawable_canvas import st_canvas

# Load the trained CNN model
model_path = "digit_recognition_model.h5"

if os.path.exists(model_path):
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("digit_recognition_model.h5")
    model = load_model()
else:
    st.error(f"Model file '{model_path}' not found. Make sure it is in the deployed directory.")

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Draw a digit below and click 'Predict'")

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280, width=280,
    drawing_mode="freedraw",
    update_streamlit=True,
    key="canvas",
)

# Function to preprocess the canvas drawing
def preprocess_canvas(canvas):
    if canvas is not None:
        img = np.array(canvas)  # Convert to NumPy array
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image
        img = img.convert("L")  # Convert RGBA to grayscale
        img = img.resize((28, 28))  # Resize to match model input size
        img = np.array(img) / 255.0  # Normalize
        img = img.reshape(1, 28, 28, 1)  # Reshape for model
        return img
    return None

# Predict when the user clicks the button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        processed_image = preprocess_canvas(canvas_result.image_data)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        st.write(f"### Predicted Digit: {predicted_digit}")
    else:
        st.write("Please draw a digit before clicking 'Predict'!")

if st.button("Clear Canvas"):
    st.rerun()  # This refreshes the Streamlit app

# Display the processed image
if canvas_result.image_data is not None:
    processed_image = preprocess_canvas(canvas_result.image_data)
    st.image(processed_image.reshape(28,28), caption="Processed Image", width=150)