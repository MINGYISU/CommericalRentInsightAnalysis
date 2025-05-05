import numpy as np
import pandas as pd
import torch
import os

# file paths
data_folder = 'processed_data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
data_path = os.path.join(data_folder, 'lease_data.pt')
scaler_path = os.path.join(data_folder, 'scalers.joblib')
checkpoint_path = 'checkpoints'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
model_path = os.path.join(checkpoint_path, f'lease_model_{len(os.listdir(checkpoint_path))}.pth')

features = [
    "time",
    "leased",
    "industry_cluster",
    "transaction_type",
    "space_type",
    "internal_class",
    "CBD_suburban",
    "zip", 
    ]

mapping = {'industry_cluster': {'Other': 0, 
                                'Technology, Advertising, Media, and Information': 1, 
                                'Financial Services and Insurance': 2, 
                                'Legal Services': 3}, 
           'transaction_type': {'New': 0, 
                                'Renewal': 1, 
                                'Expansion': 2, 
                                'TBD': 3, 
                                'Renewal and Expansion': 4, 
                                'Relocation': 5, 
                                'Extension': 6, 
                                'Restructure': 7, 
                                'Sale - Leaseback': 8}, 
           'space_type': {'New': 0, 'Relet': 1, 'Sublet': 2}, 
           'internal_class': {'O': 0, 'A': 1}, 
           'CBD_suburban': {'CBD': 0, 'Suburban': 1}, 
        }

def get_mapping(row: pd.Series):
    for feature in features:
        # if the feature needs to be mapped
        if feature in mapping:
            row[feature] = mapping[feature][row[feature]]
    return row

def normalize_features(x: torch.Tensor):
    return (x - x.mean()) / x.std()
    