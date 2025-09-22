from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

app = Flask(__name__)

# Load model and class indices
model = load_model("medicinal_plant_model_v2.h5")
with open("class_indices.json") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100
    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            predicted_class, confidence = predict_image(file_path)
            return render_template("index.html", prediction=predicted_class, confidence=confidence, image_path=file_path)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
