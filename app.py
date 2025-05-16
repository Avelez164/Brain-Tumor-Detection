import os
import random
import pickle
import logging

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, url_for
import tensorflow as tf

# ——— Flask setup ———
app = Flask(__name__)  # uses ./static by default
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— Static “NLP” summaries ———
STATIC_SUMMARY = {
    'glioma':     "Findings are consistent with a glioma tumor.",
    'meningioma': "Scan suggests a meningioma near the meninges.",
    'notumor':    "No tumor detected; MRI appears normal.",
    'pituitary':  "Evidence of a pituitary gland lesion."
}

# ——— CNN loader ———
CNN = None
def load_cnn():
    global CNN
    if CNN is not None:
        return
    here = os.path.dirname(__file__)
    # Load model JSON
    with open(os.path.join(here, 'models', 'CNN_structure.json'), 'r') as jf:
        CNN = tf.keras.models.model_from_json(jf.read())
    # Load weights
    with open(os.path.join(here, 'models', 'CNN_weights.pkl'), 'rb') as wf:
        CNN.set_weights(pickle.load(wf))
    CNN.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    logger.info("CNN loaded and compiled.")

def get_model_prediction(img_path):
    load_cnn()
    try:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        arr = np.expand_dims(np.array(img), 0) / 255.0
        preds = CNN.predict(arr)
        labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        return labels[int(np.argmax(preds[0]))]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "error"

# ——— Routes ———
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-random-image')
def get_random_image():
    try:
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        actual = random.choice(classes)
        img_dir = os.path.join(app.static_folder, 'mri-images', actual)
        img_name = random.choice(os.listdir(img_dir))
        full_path = os.path.join(img_dir, img_name)

        pred = get_model_prediction(full_path)
        summary = STATIC_SUMMARY.get(pred, "No summary available.")

        web_path = url_for('static', filename=f"mri-images/{actual}/{img_name}")
        return jsonify({
            'image_path':      web_path,
            'actual_label':    actual,
            'predicted_label': pred,
            'summary':         summary
        })
    except Exception as e:
        logger.error(f"/get-random-image error: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
