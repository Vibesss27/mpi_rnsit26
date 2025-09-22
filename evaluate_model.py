import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =======================
# CONFIGURATION
# =======================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_NAME = 'best_model_resnet50.keras'  # Make sure this matches the name of your saved model file

# =======================
# DATASET PATHS
# =======================
VAL_DIR = 'dataset/validation'  # Path to your validation dataset

# =======================
# LOAD SAVED MODEL
# =======================
if not os.path.exists(MODEL_NAME):
    print(f"Model file '{MODEL_NAME}' not found. Please make sure it is saved in the current directory.")
    exit()

print(f"Loading model: {MODEL_NAME}...")
model = load_model(MODEL_NAME)
print("\nModel loaded successfully!")

# =======================
# DATA GENERATOR
# =======================
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Do not shuffle, so we can accurately see the results
)

# =======================
# MODEL SUMMARY
# =======================
print("\nModel Architecture:")
model.summary()

# =======================
# MODEL EVALUATION
# =======================
print("\nEvaluating model on validation data...")
val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)

print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# =======================
# PREDICTIONS (OPTIONAL)
# =======================
predictions = model.predict(val_generator, verbose=1)
predicted_classes = predictions.argmax(axis=-1)
true_classes = val_generator.classes

# Plot confusion matrix (Optional)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
