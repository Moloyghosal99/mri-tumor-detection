import streamlit as st
import numpy as np
from PIL import Image
import random

CLASS_LABELS = ['Pituitary', 'Glioma', 'No Tumor', 'Meningioma']

st.title("🧠 MRI Tumor Detection")

uploaded_file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    # Fake prediction
    probs = np.random.dirichlet(np.ones(4), size=1)[0]
    idx = np.argmax(probs)

    st.success(f"Result: {CLASS_LABELS[idx]}")
    st.write(f"Confidence: {probs[idx]*100:.2f}%")

    st.bar_chart({
        CLASS_LABELS[i]: probs[i] for i in range(4)
    })
