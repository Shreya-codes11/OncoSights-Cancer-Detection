import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df_cancer = pd.read_csv('synthetic_cancer_data_v2.csv')

# Step 1: Preprocessing
# Convert categorical columns (Cancer Type and Stage) to numeric
label_encoder_cancer = LabelEncoder()
df_cancer['Cancer Type'] = label_encoder_cancer.fit_transform(df_cancer['Cancer Type'])

label_encoder_stage = LabelEncoder()
df_cancer['Stage'] = label_encoder_stage.fit_transform(df_cancer['Stage'])

# Separate features and target variable
X = df_cancer.drop('Stage', axis=1)  # Features
y = df_cancer['Stage']  # Target variable

# Normalize features (optional but recommended for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train the Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Predict and evaluate the model
y_pred = rf_model.predict(X_test)

# Output the results
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(rf_model, 'cancer_stage_model.pkl')
joblib.dump(scaler, 'scaler.pkl') 