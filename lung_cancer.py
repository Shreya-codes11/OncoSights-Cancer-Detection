import tensorflow as tf
import cv2
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('models/lung_cancer.keras')

# Define categories for prediction
categories = ["Bengin cases", "Malignant cases", "Normal cases"]

# Function to preprocess the input image
def preprocess_image(image_path, img_size=(256, 256)):
    """
    Function to preprocess the image for the model input
    :param image_path: Path to the input image
    :param img_size: Size to which the image should be resized (default is 256x256)
    :return: Preprocessed image
    """
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to the expected input size for the model
    img_resized = cv2.resize(img, img_size)
    
    # Normalize the image by scaling the pixel values to the range [0, 1]
    img_normalized = img_resized / 255.0
    
    # Expand the dimensions to fit the model's input shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
    img_array = np.repeat(img_array, 3, axis=-1)  # Convert grayscale to 3 channels
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

# Function to predict the class of an image
def predict_class(image_path):
    """
    Function to predict the lung cancer class for the given image
    :param image_path: Path to the input image for prediction
    :return: Predicted class label and prediction probabilities
    """
    img_array = preprocess_image(image_path)
    
    # Make the prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    
    # Get the predicted class label
    predicted_class = categories[predicted_class_index]
    
    return predicted_class, predictions
