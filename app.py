import os
import numpy as np
import json
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Load class labels from the JSON file
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Define confidence threshold
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for a valid prediction

# Function to preprocess the image
def preprocess_image(img_path):
    """
    Preprocess the uploaded image to match the model's requirements.
    Resizes the image to 150x150, converts to array, and normalizes pixel values.
    """
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to match the model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/')
def home():
    """
    Render the home page.
    """
    return render_template('home.html')

@app.route('/index')
def index():
    """
    Render the index page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and make predictions using the trained model.
    If the prediction confidence is low, raise an error instead of returning 'Unknown'.
    """
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded image
    file_path = os.path.join('static', 'uploaded_image.jpg')
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make prediction
    predictions = model.predict(img_array)[0]  # Get the first batch output
    confidence = np.max(predictions)           # Confidence of the top prediction
    predicted_class_index = np.argmax(predictions)  # Index of the top prediction
    predicted_label = class_labels[str(predicted_class_index)]

    # Debug: Log predictions and confidence
    print(f"Predictions: {predictions}")
    print(f"Top Confidence: {confidence}")
    print(f"Predicted Class: {predicted_label}")

    # Ensure confidence is above threshold before returning a prediction
    if confidence < CONFIDENCE_THRESHOLD:
        return render_template('result.html', prediction="Prediction confidence too low", confidence=round(confidence, 2))

    # Return the result
    return render_template(
        'result.html',
        prediction=predicted_label,
        confidence=round(confidence, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
