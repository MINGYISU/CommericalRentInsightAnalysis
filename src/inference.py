import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import os
from model import LeaseModel
from utils import get_mapping

def predict_batch(features_list):
    """
    Make predictions for multiple properties
    
    Args:
        features_list: List of dictionaries, each containing feature values
    
    Returns:
        List of predicted rent values
    """
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Apply preprocessing
    X = df.apply(get_mapping, axis=1)
    X = X.to_numpy(np.float32)
    
    # Normalize features
    X_normalized = feature_scaler.transform(X)
    
    # Convert to tensor
    X_tensor = torch.from_numpy(X_normalized).to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        
    # Denormalize predictions
    predictions = target_scaler.inverse_transform(
        outputs.cpu().numpy().reshape(-1, 1)
    ).flatten()
    
    return predictions

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model checkpoint
checkpoint_path = f'checkpoints/lease_model_{len(os.listdir("checkpoints"))-1}.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model_state_dict = checkpoint['model_state_dict']
features_order = checkpoint['features_order']
mapping = checkpoint['mapping']

# Initialize model
model = LeaseModel(len(features_order), 1).to(device)
model.load_state_dict(model_state_dict)
model.eval()  # Set to evaluation mode

# Load scalers
scaler_path = 'processed_data/scalers.joblib'
scalers = joblib.load(scaler_path)
feature_scaler = scalers['feature_scaler']
target_scaler = scalers['target_scaler']

input_features = [
    {'time': 2025.5, 'leasedSF': 1000, 'industry_cluster': 'Technology, Advertising, Media, and Information',
     'transaction_type': 'New', 'space_type': 'New', 'internal_class': 'O', 
     'CBD_suburban': 'CBD', 'zip': 95110},
    {'time': 2026, 'leasedSF': 2000, 'industry_cluster': 'Financial Services and Insurance',
     'transaction_type': 'Renewal', 'space_type': 'Relet', 'internal_class': 'A', 
     'CBD_suburban': 'Suburban', 'zip': 94105}
]
    
results = []
predicted_rents = predict_batch(input_features)
for input_feature, pred_rent in zip(input_features, predicted_rents):
    print(f"Input: {input_feature}, Predicted Rent: {pred_rent:.2f}")
    input_feature['predicted_rent'] = pred_rent
    results.append(input_feature)

# Create predictions directory if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f'predictions/predicted_rent_{len(os.listdir("predictions"))}.csv', index=False)
