import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import json

# ---------------------- GPU Memory Growth ----------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# ----------------------- Configuration -----------------------
image_size = (300, 300)
batch_size = 16
num_classes = 10  # 10 plant species
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
model_path_h5 = 'medicinal_plant_model_v2.h5'
model_path_keras = 'medicinal_plant_model_v2.keras'
csv_log = 'training_log.csv'

# ---------------------- Data Augmentation ---------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# ---------------------- Data Generators ------------------------
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# ---------------------- Model Building -----------------------
base_model = EfficientNetV2B2(include_top=False, weights='imagenet', input_shape=(300, 300, 3))
base_model.trainable = False  # Freeze base for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile base model
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------- Callbacks ------------------------
checkpoint = ModelCheckpoint(model_path_h5, monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
csv_logger = CSVLogger(csv_log)

# ---------------------- Initial Training ------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,  # Slightly reduced initial epochs to save time
    callbacks=[checkpoint, early_stop, reduce_lr, csv_logger]
)

# ------------------ Fine-Tuning (Unfreeze) ------------------
base_model.trainable = True

# Optional: freeze lower layers to speed up fine-tuning and reduce GPU memory
for layer in base_model.layers[:400]:  # freeze first 400 layers
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 10
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=fine_tune_epochs,
    callbacks=[checkpoint, early_stop, reduce_lr, csv_logger]
)

print(f"\n✅ Best model saved as {model_path_h5}")

# ------------------- Save Class Indices -------------------
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# ---------------------- Convert to .keras Format -----------------------
model = load_model(model_path_h5)
model.save(model_path_keras)
print(f"✅ Model also saved in modern format as {model_path_keras}")
