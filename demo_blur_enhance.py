# demo_blur_enhance.py

import cv2
import numpy as np
from post_training_tasks import predict_image, enhance_image

# ------------------- Configuration -------------------
sample_img_path = 'dataset/test/neem/(12).jpg'  # change path if needed
model_path = 'medicinal_plant_model_v2.h5'
class_indices_path = 'class_indices.json'
image_size = (300, 300)

# ------------------- Load Original Image -------------------
img_original = cv2.imread(sample_img_path)
if img_original is None:
    print("⚠️ Could not load image.")
    exit()

# ------------------- Apply Mild Blur -------------------
# Mild Gaussian blur to simulate low-quality image
img_blur = cv2.GaussianBlur(img_original, (5, 5), 0)  # Mild blur
blur_path = 'temp_blur.jpg'
cv2.imwrite(blur_path, img_blur)

# Display the blurred image
cv2.imshow("Blurred Image", img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------- Enhance & Predict -------------------
# Use enhancement on the blurred image
predicted_class, confidence = predict_image(
    blur_path,
    model_path,
    class_indices_path,
    image_size=image_size,
    threshold=True,
    enhance=True  # mild enhancement applied
)

print(f"\n✅ Final Prediction: {predicted_class} with confidence {confidence:.2f}%")
