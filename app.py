import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your models
age_model = tf.keras.models.load_model('age_model.h5')
gender_model = tf.keras.models.load_model('gender_model.h5')
emotion_model = tf.keras.models.load_model('emotion_model.h5')

# Define a function to make predictions
def predict(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    
    age_prediction = age_model.predict(image)
    gender_prediction = gender_model.predict(image)
    emotion_prediction = emotion_model.predict(image)
    
    age = int(age_prediction[0])
    gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"
    emotion = np.argmax(emotion_prediction)
    
    return age, gender, emotion

# Streamlit UI
st.title("AGE: Age, Gender, Emotion Detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    age, gender, emotion = predict(image)
    st.write(f"Predicted Age: {age}")
    st.write(f"Predicted Gender: {gender}")
    st.write(f"Predicted Emotion: {emotion}")
