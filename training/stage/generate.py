import pandas as pd
import numpy as np

# Function to generate synthetic dataset
def generate_cancer_data(num_samples=1000):
    # Seed for reproducibility
    np.random.seed(42)
    
    # Cancer types
    cancer_types = ['Colon', 'Lung', 'Skin']
    
    # Stages of cancer
    stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    
    # Generate random data
    data = []
    for _ in range(num_samples):
        # Random age between 20 and 80
        age = np.random.randint(20, 80)
        
        # Random tumor size (in cm)
        tumor_size = np.random.uniform(0.5, 15.0)
        
        # Random lymph node involvement (0 = No, 1 = Yes)
        lymph_node_involvement = np.random.choice([0, 1])
        
        # Random metastasis (0 = No, 1 = Yes)
        metastasis = np.random.choice([0, 1])
        
        # Random cancer type (Colon, Lung, or Skin)
        cancer_type = np.random.choice(cancer_types)
        
        # Determine stage based on tumor size, lymph node involvement, and cancer type
        if cancer_type == 'Colon':
            if tumor_size < 3.0 and lymph_node_involvement == 0 and metastasis == 0:
                stage = 'Stage 1'
            elif 3.0 <= tumor_size < 7.0:
                stage = 'Stage 2'
            elif 7.0 <= tumor_size < 10.0 or lymph_node_involvement == 1:
                stage = 'Stage 3'
            else:
                stage = 'Stage 4'
        elif cancer_type == 'Lung':
            if tumor_size < 2.5 and lymph_node_involvement == 0 and metastasis == 0:
                stage = 'Stage 1'
            elif 2.5 <= tumor_size < 5.0 or lymph_node_involvement == 1:
                stage = 'Stage 2'
            elif 5.0 <= tumor_size < 8.0 or metastasis == 1:
                stage = 'Stage 3'
            else:
                stage = 'Stage 4'
        else:  # Skin cancer
            if tumor_size < 1.0 and lymph_node_involvement == 0 and metastasis == 0:
                stage = 'Stage 1'
            elif 1.0 <= tumor_size < 3.0 or lymph_node_involvement == 1:
                stage = 'Stage 2'
            elif 3.0 <= tumor_size < 6.0 or metastasis == 1:
                stage = 'Stage 3'
            else:
                stage = 'Stage 4'
        
        # Append generated data
        data.append([age, tumor_size, lymph_node_involvement, metastasis, cancer_type, stage])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Age', 'Tumor Size (cm)', 'Lymph Node Involvement', 'Metastasis', 'Cancer Type', 'Stage'])
    
    return df

# Generate the synthetic dataset
df_cancer = generate_cancer_data(1000)

# Save to CSV
df_cancer.to_csv('synthetic_cancer_data_v2.csv', index=False)

# Show a sample of the dataset
print(df_cancer.head())
