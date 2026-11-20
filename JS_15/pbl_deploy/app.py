import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

MODEL_PATH = os.path.join(BASE_DIR, 'model_pbl.h5')
VEC_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(VEC_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("PBL Model & Vectorizer Loaded Successfully")
except Exception as e:
    print(f"Error loading PBL system: {e}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news_text = request.form.get('news_text', '').strip()
        if not news_text:
            return render_template('index.html', error='Silakan masukkan teks berita terlebih dahulu.')

        vec_input = vectorizer.transform([news_text]).toarray()
        prob = float(model.predict(vec_input, verbose=0)[0][0])

        is_valid = prob > 0.5
        label = 'Berita Valid' if is_valid else 'Berita Hoaks'
        confidence = prob if is_valid else (1.0 - prob)
        conf_pct = round(confidence * 100, 2)
        status_color = 'success' if is_valid else 'danger'

        return render_template('index.html',
                               news_text=news_text,
                               label=label,
                               confidence=conf_pct,
                               status_color=status_color,
                               prob=round(prob, 4))
    except Exception as e:
        return render_template('index.html', error=f'Terjadi kesalahan saat inferensi: {e}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
