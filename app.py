from flask import Flask, render_template, request, send_from_directory, url_for, redirect
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os, cv2, csv
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = 'models/final_crop_disease_model.h5'
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

CSV_LOG = os.path.join(RESULTS_FOLDER, "predictions_log.csv")
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Original_Image", "Heatmap_Image", "Prediction", "Confidence"])

app = Flask(__name__, static_url_path="/uploads", static_folder=UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

model = tf.keras.models.load_model(MODEL_PATH)

class_labels = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___Late_blight',
    4: 'Potato___healthy',
    5: 'Tomato_Bacterial_spot',
    6: 'Tomato_Early_blight',
    7: 'Tomato_Late_blight',
    8: 'Tomato_Leaf_Mold',
    9: 'Tomato_Septoria_leaf_spot',
    10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    11: 'Tomato__Target_Spot',
    12: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    13: 'Tomato__Tomato_mosaic_virus',
    14: 'Tomato_healthy'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_image(filepath, target_size=(224,224)):
    img = image.load_img(filepath, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

@app.route('/results/<path:filename>')
def results_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    img_array = preprocess_image(upload_path)
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    label = class_labels.get(idx, str(idx))
    conf = round(float(preds[0][idx]) * 100, 2)

    heatmap = make_gradcam_heatmap(img_array, model)
    orig = cv2.imread(upload_path)
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(orig, 0.7, heatmap_color, 0.3, 0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_name = f"{ts}_orig_{filename}"
    heat_name = f"{ts}_heat_{filename}"
    orig_path = os.path.join(app.config['RESULTS_FOLDER'], orig_name)
    heat_path = os.path.join(app.config['RESULTS_FOLDER'], heat_name)
    cv2.imwrite(orig_path, orig)
    cv2.imwrite(heat_path, blended)

    with open(CSV_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, orig_name, heat_name, label, conf])

    return render_template('result.html',
                           prediction=label,
                           confidence=conf,
                           orig_url=url_for('results_file', filename=orig_name),
                           heat_url=url_for('results_file', filename=heat_name))

@app.route('/gallery')
@app.route('/gallery/<int:page>')
def gallery(page=1):
    per_page = 10
    entries = []
    if os.path.exists(CSV_LOG):
        with open(CSV_LOG, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)

    entries.reverse()

    total_entries = len(entries)
    total_pages = (total_entries + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_entries = entries[start:end]

    return render_template(
        'gallery.html',
        entries=paginated_entries,
        page=page,
        total_pages=total_pages
    )


if __name__ == '__main__':
    app.run(debug=True)
