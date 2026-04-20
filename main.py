from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import gdown

# -------------------- APP SETUP --------------------
app = Flask(__name__)

# -------------------- DOWNLOAD MODEL --------------------
MODEL_PATH = 'models/model.h5'

if not os.path.exists(MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    url = "https://drive.google.com/uc?id=1Vv63je3EneQDFe-o8LCLxd7jKpyxxMuR"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------- LOAD MODEL --------------------
model = load_model(MODEL_PATH, compile=False)
# -------------------- CONSTANTS --------------------
CLASS_LABELS = ['Pituitary', 'Glioma', 'No Tumor', 'Meningioma']

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# -------------------- PREDICTION FUNCTION --------------------
def predict_tumor(image_path):
    IMAGE_SIZE = 128

    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probs = predictions[0]

    predicted_index = np.argmax(probs)
    confidence = float(probs[predicted_index])
    tumor_type = CLASS_LABELS[predicted_index]

    # RESULT LOGIC
    if tumor_type == 'No Tumor':
        result = "No Tumor Detected"
        risk = "Low"
        advice = "No tumor detected. Stay healthy!"
    else:
        result = f"Tumor: {tumor_type}"

        if confidence > 0.8:
            risk = "High"
            advice = "Consult a neurologist immediately."
        elif confidence > 0.5:
            risk = "Medium"
            advice = "Further medical tests are recommended."
        else:
            risk = "Low"
            advice = "Monitor symptoms and consult doctor if needed."

    # WARNING LOGIC
    if confidence > 0.85:
        warning = "Prediction is highly confident, but not 100% medically reliable."
    elif confidence > 0.6:
        warning = "⚠️ Moderate confidence - result should be verified by a doctor."
    else:
        warning = "⚠️ Low confidence - prediction may be incorrect."

    return result, confidence, risk, advice, probs.tolist(), warning


# -------------------- ROUTES --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            result, confidence, risk, advice, probs, warning = predict_tumor(file_path)

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence * 100:.2f}%",
                risk=risk,
                advice=advice,
                probs=probs,
                warning=warning,
                file_path=f'/uploads/{file.filename}'
            )

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# -------------------- RUN APP --------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # 🔥 IMPORTANT FOR RENDER
    app.run(host='0.0.0.0', port=port)
