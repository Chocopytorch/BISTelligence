import pandas as pd
from BackEnd.scripts.load import load_data
from sklearn.preprocessing import MinMaxScaler

def PreprocessData(path:str):
    '''
    @params
    path : dataset path
    
    @logic
    load model ->
    preprocess data
    '''
    
    data = load_data(path)
    
    prev_sum=data[data['key']==4].loc[942:981]['Vibration_RMS2'].sum()
    after_sum=data[data['key']==4].loc[983:1022]['Vibration_RMS2'].sum()
    aver=(prev_sum+after_sum)/80
    data.loc[data['Vibration_RMS2']<0,'Vibration_RMS2']=aver
    
    return data

def ScalingData(data):
    '''
    @params
    data : dataset
    
    @logic
    Minmax Scaling ->
    column named key re-add
    '''
    
    scaler=MinMaxScaler()
    scaled_data=pd.DataFrame(scaler.fit_transform(data.drop(['key'],axis=1)),columns=data.columns[1:])
    
    # key 값을 다시 추가
    scaled_data['key'] = data['key']
    
    exclude_columns = ['key']
    data_columns = scaled_data.drop(columns=exclude_columns)
    data_without_key = pd.DataFrame(scaled_data, columns=data_columns.columns)
    
    return scaled_data, data_without_key