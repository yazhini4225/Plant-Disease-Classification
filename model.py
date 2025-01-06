import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Preprocess the image
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    if img_array.shape[-1] != 3:  # Ensure 3 channels (RGB)
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Perform prediction
CLASS_NAMES = ['Healthy', 'Diseased']  # Replace with your class labels

def model_prediction(img: Image.Image):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return CLASS_NAMES[predicted_class.item()]
