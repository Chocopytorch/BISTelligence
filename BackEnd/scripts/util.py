import os
import numpy as np
import matplotlib.pyplot as plt
import BackEnd.scripts.preprocess as sp
import BackEnd.model.model as mm
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn

class Util():
  def __init__(self):
    self.path = os.path.join(os.path.dirname(__file__), '../src/data.csv')
    self.data = None
    self.data_exc_key = None
    self.model = None
    
  def SetData(self):
    self.data, self.data_exc_key = sp.ScalingData(sp.PreprocessData(self.path))
    
    return self.data, self.data_exc_key
    
  def GetModel(self, model_name:str, param_dict:dict):
    if model_name == 'Lof':
        self.model = mm.BaseModel.GetLocalOutlierFactor(param_dict)
    else:
        self.model = mm.BaseModel.GetVQVAE(param_dict)

    return self.model
  
  def Predict(self, data, data_exc_key, model_name:str, model):
    # Detect key change points and store indices of changes
    key_change_indices = data.index[data['key'].diff() != 0].tolist()

    # Adjust to capture the previous index (the one before the change)
    key_change_indices = [data.index[i - 1] for i in range(1, len(data.index)) if data['key'].iloc[i] != data['key'].iloc[i - 1]]

    # Remove index 0 if it's in the list
    key_change_indices = [idx for idx in key_change_indices if idx != 0]

    # Add the last index to key_change_indices
    key_change_indices.append(data.index[-1])
    
    if model_name == 'Lof':
        model.fit(data_exc_key)

        # Get the anomaly scores (negative outlier factor)
        score = model.negative_outlier_factor_

        # The anomaly scores are negative, so we can convert them to positive values to indicate anomaly
        # The more negative the score, the more anomalous the data
        score = -score  # Invert to make the score positive (more positive = more anomalous)

        # Use fit_predict to get predictions on training data
        y_pred = model.fit_predict(data_exc_key)
        
        return score, y_pred, key_change_indices
    else:
        # Predict Anomalies on m_df (same as before)
        model.eval()
        X_test_tensor = torch.tensor(data_exc_key.values, dtype=torch.float32)
        with torch.no_grad():
            recon_x, z_e = model(X_test_tensor)

        # Calculate anomaly scores using reconstruction error (MSE)
        y_pred = torch.mean((X_test_tensor - recon_x) ** 2, dim=1).numpy()

        # Slice anomaly_scores to match the length of m_df.index[start_index:]
        score = y_pred[:len(data.index[0:])]
        
        return score, y_pred, key_change_indices