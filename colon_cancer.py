9+import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('models/colon_cancer.keras')  # Provide the correct path

# Define the class labels (you should modify this if your labels are different)
class_labels = ['Colon Adenocarcinoma', 'Colon Benign Tissue']

# Function to preprocess an image
def preprocess_image(img_path):
    """
    Function to preprocess the image by resizing and normalizing it.
    :param img_path: Path to the image
    :return: Preprocessed image ready for prediction
    """
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to match model's expected input
    img_array = image.img_to_array(img)  # Convert image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image to [0, 1]
    return img_array

# Function to make predictions
def predict_cancer(img_path):
    """
    Function to predict the type of cancer (colon adenocarcinoma or colon benign tissue).
    :param img_path: Path to the image for prediction
    :return: Predicted cancer type label
    """
    img_array = preprocess_image(img_path)
    
    # Make the prediction
    prediction = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = class_labels[predicted_class[0]]
    
    return predicted_label
