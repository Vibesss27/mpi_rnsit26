import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import pandas as pd

# ----------------------- Configuration -----------------------
model_path = 'medicinal_plant_model_v2.h5'
class_indices_path = 'class_indices.json'
image_size = (300, 300)
sample_img_path = 'dataset/test/neem/(12).jpg'  # change path if needed
validation_dir = 'dataset/validation'
max_images_per_class = 100  # LIMIT for speed

# ------------------- Image Enhancement Function -------------------
def enhance_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("⚠️ Could not load image for enhancement:", img_path)
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_CUBIC)
    img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y_channel = img_y_cr_cb[:, :, 0]
    y_channel = cv2.equalizeHist(y_channel)
    img_y_cr_cb[:, :, 0] = y_channel
    img = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2RGB)
    return img

# ------------------- Prediction Function -------------------
def predict_image(img_path, model_path, class_indices_path, image_size=(300, 300), threshold=True, enhance=True):
    model = load_model(model_path)
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}

    if enhance:
        img_cv = enhance_image(img_path)
        if img_cv is None:
            return None, 0
        img_array = img_cv / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    else:
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_cv = cv2.imread(img_path)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    if threshold:
        if confidence >= 50:
            confidence_msg = "✅ Reliable prediction"
        elif 40 <= confidence < 50:
            confidence_msg = "⚠️ Can attempt prediction, not fully reliable"
        else:
            confidence_msg = "❌ Prediction not reliable"
    else:
        confidence_msg = ""

    print(f"🔍 Predicted Class: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}% {confidence_msg}")

    return predicted_class, confidence

# ------------------- Plot Training History -------------------
def plot_training_log(csv_log_file='training_log.csv'):
    if not os.path.exists(csv_log_file):
        print(f"⚠️ {csv_log_file} not found. Cannot plot training history.")
        return
    history = pd.read_csv(csv_log_file)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------- Compute Metrics (Fast: limited images) -------------------
def compute_metrics_fast(model_path, validation_dir, class_indices_path, image_size=(300, 300), max_images_per_class=100):
    model = load_model(model_path)
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}

    y_true = []
    y_pred = []

    for class_name in os.listdir(validation_dir):
        class_path = os.path.join(validation_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        img_files = os.listdir(class_path)[:max_images_per_class]
        for img_file in img_files:
            img_path = os.path.join(class_path, img_file)
            img = enhance_image(img_path)
            if img is None:
                continue
            img_array = img / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            y_pred.append(predicted_index)
            y_true.append(class_indices[class_name])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels.values(),
                yticklabels=class_labels.values(), cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (First {max_images_per_class} images per class)')
    plt.show()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_labels.values())
    print("\n📄 Classification Report:\n")
    print(report)

    # Precision, Recall, F1 per class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    print("\n📊 Precision, Recall, F1 per class:")
    for idx, cls in enumerate(class_labels.values()):
        print(f"{cls} -> Precision: {precision[idx]:.2f}, Recall: {recall[idx]:.2f}, F1: {f1[idx]:.2f}")

# ------------------- Run Fast Tasks -------------------
if __name__ == "__main__":
    plot_training_log()
    predict_image(sample_img_path, model_path, class_indices_path, image_size)
    compute_metrics_fast(model_path, validation_dir, class_indices_path, image_size, max_images_per_class)
