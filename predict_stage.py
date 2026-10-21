import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def predict_cancer_stage(age, tumor_size, lymph_node_involvement, metastasis, cancer_type):
    try:
        # Load the trained model and scaler
        model = joblib.load('models/cancer_stage_model.pkl')
        scaler = joblib.load('models/scaler.pkl')  # Load the saved scaler

        # Load label encoders for cancer type and stage
        label_encoder_cancer = LabelEncoder()
        label_encoder_stage = LabelEncoder()

        # Load the synthetic dataset for encoder reference (ensure this file exists)
        df_cancer = pd.read_csv('models/synthetic_cancer_data_v2.csv')

        # Fit label encoders with the data
        label_encoder_cancer.fit(df_cancer['Cancer Type'])
        label_encoder_stage.fit(df_cancer['Stage'])

        # Validate and encode cancer_type
        if not cancer_type or cancer_type not in label_encoder_cancer.classes_:
            raise ValueError(f"Invalid cancer type: '{cancer_type}'. Please provide a valid type.")

        cancer_type_encoded = label_encoder_cancer.transform([cancer_type])[0]

        # Preprocess the input data
        input_data = np.array([[age, tumor_size, lymph_node_involvement, metastasis, cancer_type_encoded]])

        # Normalize the features using the same scaler used during training
        input_data_scaled = scaler.transform(input_data)

        # Predict the cancer stage
        predicted_stage_encoded = model.predict(input_data_scaled)[0]

        # Decode the predicted stage back to original
        predicted_stage = label_encoder_stage.inverse_transform([predicted_stage_encoded])[0]

        return predicted_stage

    except ValueError as e:
        return f"Input Error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
