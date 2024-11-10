import BackEnd.scripts.util as su
import BackEnd.scripts.load as sl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os

def train_model(modelname:str, params:dict, trainable:bool):
    # 데이터 준비
    se_class = su.Util()
    data, data_exc_key = se_class.SetData()
    
    # 모델 준비
    model = se_class.GetModel(model_name=modelname, param_dict=params)
    
    if trainable:
        # 모델을 저장할 디렉터리
        save_path = os.path.join(os.path.dirname(__file__), '../model/saved_model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if modelname == 'Lof':
            model.fit(data_exc_key)
        else:
            window_size = 25  # Example window size
            X_train = np.array([data_exc_key[i:i + window_size] for i in range(len(data_exc_key) - window_size)])
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training loop
            epochs = 300
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                recon_x, z_e = model(X_train_tensor)
                loss = criterion(recon_x, X_train_tensor)
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
                    
        print(f'Finish Fitting {modelname} model!')
        
        # 모델을 지정된 경로에 저장
        model_file = os.path.join(save_path, f"{modelname}_model.pkl")
        joblib.dump(model, model_file)
        
        print(f"Model saved at {model_file}")
        
        model = sl.load_model(modelname=modelname)
    else:
        model = sl.load_model(modelname=modelname)
        
    return data, data_exc_key, model