import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from flask import Flask, request, render_template_string
from skimage.feature import hog

app = Flask(__name__)

# Load Model & Scaler
MODEL_PATH = 'day_night_model.h5'
SCALER_PATH = 'scaler.pkl'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("System Loaded Successfully")
except Exception as e:
    print(f"Error loading system: {e}")

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray, orientations=9, pixels_per_cell=(8,8),
        cells_per_block=(2,2), block_norm='L2-Hys',
        visualize=False, feature_vector=True
    )
    return scaler.transform(hog_feat.reshape(1, -1))

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Day vs Night Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f6f9; display: flex; justify-content: center; align-items: center; min-height: 80vh; }
            .card { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; max-width: 450px; width: 100%; }
            h1 { color: #2c3e50; font-size: 24px; margin-bottom: 20px; }
            input[type="file"] { margin: 20px 0; padding: 10px; border: 1px dashed #3498db; border-radius: 6px; width: 80%; }
            button { background-color: #3498db; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; font-weight: 600; }
            button:hover { background-color: #2980b9; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Day vs Night Classifier</h1>
            <p>Unggah citra untuk mengklasifikasikan siang atau malam.</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <button type="submit">Prediksi</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        data = preprocess_image(file.read())
        prediction = float(model.predict(data, verbose=0)[0][0])

        is_day = prediction > 0.5
        label = "Day (Siang)" if is_day else "Night (Malam)"
        confidence = prediction if is_day else (1.0 - prediction)
        badge_color = "#27ae60" if is_day else "#2c3e50"

        return f'''
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8">
            <title>Hasil Prediksi</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f6f9; display: flex; justify-content: center; align-items: center; min-height: 80vh; }}
                .card {{ background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; max-width: 450px; width: 100%; }}
                .badge {{ display: inline-block; padding: 8px 18px; border-radius: 20px; color: white; background: {badge_color}; font-size: 20px; font-weight: bold; margin: 15px 0; }}
                a {{ display: inline-block; margin-top: 20px; color: #3498db; text-decoration: none; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h2>Hasil Klasifikasi</h2>
                <div class="badge">{label}</div>
                <p>Confidence: {confidence*100:.2f}%</p>
                <p>Sigmoid: {prediction:.4f}</p>
                <a href="/">Kembali</a>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
