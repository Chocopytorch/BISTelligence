import joblib
import os
import pandas as pd
from PIL import Image
import io

def load_model(modelname:str):
    # 모델을 로드할 디렉터리
    load_path = os.path.join(os.path.dirname(__file__), '../model/saved_model')
    
    model_file = os.path.join(load_path, f"{modelname}_model.pkl")
    model = joblib.load(model_file)
    
    print(f"Model Loaded from {model_file} successfully!")
    
    return model

def load_data(path):
    data=pd.read_csv(path)
    
    return data

def load_png(modelname):
    load_path = os.path.join(os.path.dirname(__file__), '../model/logs')
    path = os.path.join(load_path, f'anomaly_detection_plot_{modelname}.png')
    
    # 이미지 파일 열기
    with Image.open(path) as img:
        # 이미지를 바이트 스트림으로 변환
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr