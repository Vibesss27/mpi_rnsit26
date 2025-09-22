import tensorflow as tf

# Path to the saved model
model_path = 'best_model_resnet50.keras'

try:
    # Load the model from the saved file
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    # Print a summary of the model to check its structure
    model.summary()

except Exception as e:
    print(f"Error loading the model: {e}")
