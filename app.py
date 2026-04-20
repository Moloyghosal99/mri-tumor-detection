import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import gdown

# ---------------- MODEL DOWNLOAD ----------------
MODEL_PATH = 'model.h5'

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1Vv63je3EneQDFe-o8LCLxd7jKpyxxMuR"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# ---------------- CLASS LABELS ----------------
CLASS_LABELS = ['Pituitary', 'Glioma', 'No Tumor', 'Meningioma']

# ---------------- UI ----------------
st.title("🧠 MRI Tumor Detection")
st.write("Upload an MRI image to detect tumor")

uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Preprocess
    img = load_img("temp.jpg", target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    probs = predictions[0]

    predicted_index = np.argmax(probs)
    confidence = probs[predicted_index]

    result = CLASS_LABELS[predicted_index]

    st.subheader(f"🧠 Result: {result}")
    st.write(f"📊 Confidence: {confidence*100:.2f}%")

    # Chart
    st.bar_chart({
        "Pituitary": probs[0],
        "Glioma": probs[1],
        "No Tumor": probs[2],
        "Meningioma": probs[3]
    })
