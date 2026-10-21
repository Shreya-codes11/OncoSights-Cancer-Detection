import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/skin_cancer.h5')  # Replace with the correct path to your model file

# Function to preprocess input image
def preprocess_image(image_path):
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or invalid format")
        
        # Resize the image to the input shape of the model (32x32x3)
        img_resized = cv2.resize(img, (32, 32))
        
        # Normalize the image (if required by the model training process)
        img_normalized = img_resized / 255.0
        
        # Reshape the image to (1, 32, 32, 3)
        img_reshaped = np.expand_dims(img_normalized, axis=0)
        
        return img_reshaped
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Function to predict the class of the image
def predict_class(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    if img is None:
        return "Error: Could not preprocess image."
    
        # Predict the class of the image
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)  # Get the class with the highest probability

    # Define cancer classes with stages
    cancer_classes = {
        0: 'Melanoma',
        1: 'Basal Cell Carcinoma',
        2: 'Squamous Cell Carcinoma'
    }

    # Get cancer type and stage based on predicted class
    predicted_cancer_type = cancer_classes.get(predicted_class[0], "Unknown")
    cancer_presence = "Present" if predicted_class[0] in cancer_classes else "Absent"
    return predicted_class,cancer_presence
