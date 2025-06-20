import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle

# === Load model ===
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "skin_model.h5")
model = tf.keras.models.load_model(model_path)

# === Class names ===
class_names = ['melanoma', 'nevus', 'seborrheic_keratosis']

# === Streamlit config ===
st.set_page_config(page_title="Skin Cancer Detection", layout="wide")
st.title("🔬 Skin Cancer Detection App")
st.markdown("Upload a skin lesion image to detect the type of skin cancer.")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image_data = Image.open(uploaded_file).convert("RGB")
        st.image(image_data, caption="🖼 Uploaded Image", use_container_width=True)

    with col2:
        image_resized = image_data.resize((128, 128))
        image_array = np.array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        confidence = np.max(prediction) * 100
        predicted_class = class_names[np.argmax(prediction)]

        st.markdown("## 🧠 Prediction Result")
        st.success(f"🎯 **Predicted Class**: `{predicted_class.upper()}`")
        st.info(f"🔎 **Confidence**: `{confidence:.2f}%`")

        st.markdown("### 📊 Class Probabilities")
        for i, cls in enumerate(class_names):
            st.write(f"- {cls.title()}: `{prediction[0][i]*100:.2f}%`")

# === Accuracy/Loss Plot ===
history_path = os.path.join(os.path.dirname(__file__), "..", "model", "history.pkl")

if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    st.markdown("## 📈 Training History")

    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='Train Accuracy')
    ax.plot(history['val_accuracy'], label='Val Accuracy')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    st.pyplot(fig)

# === About Section ===
with st.expander("ℹ️ About this Project"):
    st.markdown("""
    This is a final year B.Tech project developed using a Convolutional Neural Network (CNN) for skin cancer detection.
    
    - **Model:** TensorFlow / Keras
    - **Interface:** Streamlit
    - **Classes:** Melanoma, Nevus, Seborrheic Keratosis
    - **Dataset:** ISIC (International Skin Imaging Collaboration)
    - **Developers:** TANISH (210316), MEHVISH NABI (210361), TAJAMUL HUDDA (210364), ANAMUL HAQ WAR (210342)
    """)
