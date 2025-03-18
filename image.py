import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model("dog_cat_model (1).h5")

# Upload an image
uploaded_file = st.file_uploader("Upload an image...", type=None)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocessing
    img = img.resize((256, 256))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    
    prediction = model.predict(img_array)

    
    class_label = "Dog" if prediction[0] > 0.5 else "Cat"
    st.write(f"### Prediction: {class_label}")
