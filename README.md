# OncoSights – Cancer Detection and Prediction System

OncoSights is an AI-based cancer detection and prediction system designed to assist in the preliminary analysis of lung, skin, and colon cancer. The system analyzes uploaded medical images and provides cancer classification and stage prediction based on the available model inputs.

The project aims to make AI-assisted cancer analysis more accessible through a simple web interface, along with information about precautions, treatment options, and medical terminology.

## Key Features

- **Cancer Image Classification:** Supports lung, skin, and colon cancer image analysis.
- **Image Upload:** Allows users to upload supported medical images.
- **Cancer Stage Prediction:** Predicts stages based on inputs such as age, tumor size, lymph-node involvement, and metastasis, according to the implemented model.
- **Precautions and Treatment Information:** Displays relevant information to help users understand possible next steps.
- **Web-Based Interface:** Provides an accessible interface built using Flask and Tailwind CSS.
- **AI Health Chatbot:** A Gemini-powered chatbot can be integrated to explain medical terminology and answer general health-related questions when API configuration is available.

## Cancer Types Supported

1. **Lung Cancer**
2. **Skin Cancer**
3. **Colon Cancer**

## Technologies Used

- **Programming Language:** Python
- **Backend Framework:** Flask
- **Frontend:** HTML, Tailwind CSS
- **Machine Learning / Deep Learning:** TensorFlow, Keras, Scikit-learn
- **Model Storage:** `.keras`, `.h5`, and `.pkl` files
- **Development Tools:** Visual Studio Code, Git, GitHub

*The exact libraries required depend on the models and implementation included in the repository.*

## Project Structure

```text
OncoSights/
├── app.py
├── templates/
│   └── cancer.html
├── static/
│   └── home.jpg
├── models/
│   ├── cancer_classifier.h5
│   ├── lung_cancer.keras
│   ├── colon_cancer.keras
│   ├── skin_cancer.h5
│   └── cancer_stage_model.pkl
├── training/
│   ├── lung/
│   └── colon/
├── images.zip
├── .gitignore
└── README.md
```

*This is an illustrative structure. Adjust the names to match the actual files in your repository.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shreya-codes11/OncoSights-Cancer-Detection.git
cd OncoSights-Cancer-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the project using the virtual environment's Python executable directly.

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the packages required by the project. For example:

```bash
pip install flask tensorflow scikit-learn pillow numpy
```

Additional dependencies may be needed depending on the imports in `app.py` and the model files.

### 4. Add the trained model files

Large trained models may be excluded from GitHub because individual files exceed GitHub's normal 100 MB file limit.

Place the required model files in the `models/` directory, following the filenames expected by the application. Obtain them from a trusted source or arrange access to them separately.

**Note:** The application will not perform all predictions until the required model files are available.

### 5. Run the application

```bash
python app.py
```

Open the local URL displayed in the terminal, commonly:

```text
http://127.0.0.1:5000/
```

## How to Use

1. Start the Flask application.
2. Open the website in your browser.
3. Select the supported cancer-analysis option.
4. Upload a supported medical image.
5. Review the classification result, if the required model is available.
6. Enter the required clinical inputs for stage prediction, where applicable.
7. Review the displayed precautions and treatment information.

The exact workflow depends on the features implemented in the current version.

## Model and Performance Notes

OncoSights is designed to use trained machine-learning and deep-learning models for image classification and stage prediction.

If reporting a model accuracy such as **97%**, specify the dataset, evaluation method, and test results that support the figure. Accuracy may vary across datasets and real-world medical images.

## Important Medical Disclaimer

OncoSights is an academic project intended for educational and research purposes. Its predictions are not a substitute for professional medical advice, diagnosis, or treatment. Results should not be used as the sole basis for clinical decisions. Consult a qualified healthcare professional for medical evaluation and treatment recommendations.

## Security and Privacy

- Do not commit API keys, passwords, or other secrets to GitHub.
- Store any required API credentials in environment variables.
- Avoid uploading identifiable patient records or private medical images to public repositories.
- Only load trusted model files, particularly serialized `.pkl` files.

## Future Enhancements

- Improve model evaluation using independent test datasets.
- Enhance the user interface and prediction-result presentation.
- Add clearer model-performance and limitation reporting.
- Improve deployment and model-file management.
- Expand patient-friendly explanations and educational resources.

## Author

**Shreya Shinde**

GitHub: [@Shreya-codes11](https://github.com/Shreya-codes11)

---

*OncoSights – AI-assisted insights for cancer analysis and prediction.*
