import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from cancer_classifier import predict_image_type  # Import the function from cancer_classifier.py
from colon_cancer import predict_cancer  # Import the function from colon_cancer.py
from lung_cancer import predict_class as predict_lung_class  # Import lung cancer prediction
from skin_cancer import predict_class as predict_skin_class  # Import skin cancer prediction
from predict_stage import predict_cancer_stage  # Import the predict_cancer_stage function

# Initialize Flask app
app = Flask(__name__)

# Configure the Gemini API
genai.configure(api_key="")

img_class = ""
# Set up upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure app to use upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cancerpage')
def cancerpage():
    return render_template('cancer.html')

@app.route('/healthbot')
def healthbot():
    return render_template('chatbot.html')

@app.route('/adminlogin')
def adminlogin():
    return render_template('admin_login.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Classify the uploaded image using the cancer classifier model
        predicted_class = predict_image_type(file_path)  # Using the imported function to classify the image type

        # Initialize variables to store cancer status and class
        cancer_present = 'Absent'
        cancer_class = ''

        # Determine the cancer type and predict using respective models
        if predicted_class == 'colon':
            predicted_class_result = predict_cancer(file_path)
            cancer_present = 'Present' if 'Adenocarcinoma' in predicted_class_result else 'Absent'
            cancer_class = predicted_class_result
        elif predicted_class == 'lung':
            predicted_class_result, _ = predict_lung_class(file_path)
            cancer_present = 'Present' if predicted_class_result == 'Malignant cases' else 'Absent'
            cancer_class = predicted_class_result
        elif predicted_class == 'skin':
            predicted_class_result,cancer_present = predict_skin_class(file_path)
              # Define cancer classes with stages
            cancer_classes = {
                0: 'Melanoma',
                1: 'Basal Cell Carcinoma',
                2: 'Squamous Cell Carcinoma',
                3: 'Vascular Lesion',
                4: 'Benign Keratosis',
                5: 'Dermatofibroma',
                6: 'Non-cancerous'
            }
            cancer_class = cancer_classes[int(str(predicted_class_result)[1:-1])]
        else:
            predicted_class_result = 'Unknown'
            cancer_class = 'N/A'

        img_class = predicted_class
        # Return the classification result
        return jsonify({
            'message': 'Image successfully uploaded and classified!',
            'file_path': file_path,
            'image_type':  f'Image: {predicted_class}',  # Send the image type (lung, skin, colon, other)
            'cancer_status': f'Cancer: {cancer_present}',
            'cancer_class': f'Class: {cancer_class}',  # Show cancer class if present
        })
    else:
        return jsonify({'error': 'Invalid file format'})

@app.route('/predict', methods=['POST'])
def predict_stage_route():
    try:
        # Retrieve form data
        age = int(request.form.get('age', 1))  # Default to 1 if age is not provided
        tumor_size = float(request.form.get('tumor_size', 0.0))  # Default to 0.0 if tumor_size is not provided
        lymph_node_involvement = int(request.form.get('lymph_node_involvement', 0))  # Default to 0 if not provided
        metastasis = int(request.form.get('metastasis', 0))  # Default to 0 if not provided
        cancer_type = request.form.get('cancer_type', '').replace('Image:', '').strip().capitalize()

        # Validate cancer_type
        if not cancer_type:
            return jsonify({'error': 'Cancer type is required.'}), 400

        # Predict the cancer stage
        predicted_stage = predict_cancer_stage(age, tumor_size, lymph_node_involvement, metastasis, cancer_type)
        
        # Return the predicted stage
        return jsonify({
            'success': True,
            'cancer_type': cancer_type,
            'predicted_stage': f'Predicted Cancer Stage: {predicted_stage}',
        })

    except ValueError as ve:
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# Function to get response from Gemini model
def get_chatbot_response(user_input):
    model = genai.GenerativeModel("gemini-1.5-flash")
    user_input = "Please give response of following in points do not give paragraphs. Do not give any unnecessary decorations and text. also give formatted output : " + user_input
    response = model.generate_content(user_input)
    return response.text

# Route to render chatbot page
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        # Get user input from form
        user_input = request.form['user_input']
        # Get chatbot response
        response_text = get_chatbot_response(user_input)
        return jsonify({'response': response_text})
    # Render the initial chatbot page
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
