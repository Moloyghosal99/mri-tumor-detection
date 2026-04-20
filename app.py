import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown
import os

MODEL_PATH = "model.h5"

# Download model
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1Vv63je3EneQDFe-o8LCLxd7jKpyxxMuR"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

CLASS_LABELS = ['Pituitary', 'Glioma', 'No Tumor', 'Meningioma']

def predict(image):
    img = image.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    return {
        CLASS_LABELS[i]: float(preds[i]) for i in range(len(CLASS_LABELS))
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="🧠 MRI Tumor Detection"
)

demo.launch()
