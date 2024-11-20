import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
import numpy as np

# Add background color using markdown and CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #ADD8E6;  /* Light Blue background */
        }
    </style>
""", unsafe_allow_html=True)

# Display title and description
st.title("Emotion Detector App")
st.write("This app predicts the emotion (Happy or Sad) from images that you upload. Please upload an image below to get started.")

# Upload file section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load and process the image
if uploaded_file is not None:
    # Load and display the image
    img = load_img(uploaded_file, target_size=(200, 200))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    # Create model and load weights
    def create_model():
        model = Sequential([
            Input(shape=(200, 200, 3)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPool2D(2, 2),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPool2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        model.load_weights('emotion_detector.weights.h5')  # Load the saved model weights
        return model

    model = create_model()

    # Predict emotion
    prediction = model.predict(img_array)
    if prediction < 0.5:
        st.write("**Prediction: Happy 😊**")
    else:
        st.write("**Prediction: Sad 😔**")