import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('brain_tumor_model.h5')

def preprocess_image(image):
    """Function to preprocess the image: resizing and normalization."""
    image = image.resize((128, 128))
    image = np.array(image) 
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0) 
    return image

st.title("Brain Tumor Detection")
uploaded_file = st.file_uploader("Upload brain scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_class = classes[np.argmax(prediction)]

    if predicted_class == 'notumor':
        st.write("No brain tumor detected.")
    else:
        st.write(f"Detected class: {predicted_class}")

