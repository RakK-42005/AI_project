import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('model/currency_model.h5')

# Define class names (as per folders)
class_names = ['10', '20', '50', '100', '500', '2000']

st.title("💰 Indian Currency Note Detection")
st.write("Upload an image of a currency note to detect its denomination.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.success(f"💵 Detected Note: ₹{result}")
    st.info(f"Confidence: {confidence:.2f}%")
