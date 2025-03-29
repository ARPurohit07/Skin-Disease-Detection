import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time  

model = load_model("skin_disease_model.h5")

# Define disease classes
disease_classes = ['Melanoma', 'Nevus', 'Basal Cell Carcinoma',
                   'Actinic Keratosis', 'Benign Keratosis',
                   'Dermatofibroma', 'Vascular Lesion']

def preprocess_image(image):
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

st.set_page_config(page_title="Skin Disease Detector", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
        .stTitle { text-align: center; font-size: 32px; color: #EAECEE; }
        .stTextInput > label, .stFileUploader > label { font-weight: bold; color: #AAB7B8; }
        .uploadedImage { border-radius: 12px; box-shadow: 0px 4px 10px rgba(0,0,0,0.2); }
        .predictionCard {
            padding: 15px; 
            border-radius: 12px; 
            background: rgba(44, 62, 80, 0.85); /* Dark semi-transparent */
            color: #ECF0F1; 
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='stTitle'> Skin Disease Detection AI</h1>", unsafe_allow_html=True)
st.write("Upload a skin lesion image, and AI will predict the disease category.")

st.markdown("---")

st.subheader("📤 Upload Images")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

if not st.session_state.clear_flag:
    uploaded_files = st.file_uploader("Select one or more images:", type=["jpg", "png", "jpeg"],
                                      accept_multiple_files=True, key="uploader")
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
else:
    st.session_state.uploaded_files = []
    st.session_state.clear_flag = False

st.markdown("---")

col1, col2 = st.columns([1, 1])
predict_clicked = col1.button("🔍 Predict", use_container_width=True)
clear_clicked = col2.button("❌ Clear Images", use_container_width=True)

if clear_clicked:
    st.session_state.uploaded_files = []
    st.session_state.clear_flag = True
    st.rerun()

if st.session_state.uploaded_files and predict_clicked:
    st.subheader("📊 Predictions")
    for uploaded_file in st.session_state.uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(Image.open(uploaded_file), caption=f"Uploaded: {uploaded_file.name}",
                     use_container_width=True, output_format="JPEG", clamp=True)

        with st.spinner("Analyzing image..."):
            time.sleep(1.5)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100  # Confidence score

        with col2:
            st.markdown(f"""
            <div class='predictionCard'>
                <h4>🧬 Prediction for <strong>{uploaded_file.name}</strong></h4>
                <p><b>Disease:</b> {disease_classes[predicted_class]}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.info("💡 This AI model helps with preliminary analysis but is not a replacement for a doctor. Always consult a dermatologist for an accurate diagnosis.")
footer_style = """
    <style>
        .footer {
            position: relative; /* Instead of fixed, to prevent overlap */
            width: 100%;
            background: linear-gradient(to right, #2C3E50, #34495E); /* Dark gradient */
            text-align: center;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
            color: #ECF0F1; /* Light text for contrast */
            border-top: 2px solid #1A252F; /* Subtle border */
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.3); /* Soft shadow */
            margin-top: 30px; /* Adds space before the footer */
            border-radius: 8px 8px 0 0; /* Soft curved edges at the top */
        }
        .footer a {
            text-decoration: none;
            color: #00A6FF; /* Bright blue for contrast */
            font-weight: bold;
            transition: color 0.3s ease-in-out;
        }
        .footer a:hover {
            color: #FFA500; /* Orange hover effect */
        }
    </style>
    <div class="footer">
        Developed by  
        <a href="https://github.com/ARPurohit07" target="_blank">ARPurohit07</a>
    </div>
"""

st.markdown(footer_style, unsafe_allow_html=True)
