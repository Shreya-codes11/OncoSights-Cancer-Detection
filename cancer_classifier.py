import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained cancer classifier model
model = tf.keras.models.load_model('models/cancer_classifier.h5')

# Define image size
image_size = (224, 224)

def predict_image_type(img_path):
    """
    Function to predict the cancer image type (colon, lung, or skin).
    :param img_path: Path to the input image
    :return: Predicted class label
    """
    # Load the image with the required target size
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Get predictions from the model
    predictions = model.predict(img_array)
    
    # Class labels for the prediction (colon, lung, skin)
    class_labels = ['colon', 'lung', 'skin']
    
    # Get the index of the highest prediction
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Return the predicted class label
    predicted_class = class_labels[predicted_class_index]
    return predicted_class
