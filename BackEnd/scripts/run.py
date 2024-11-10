import BackEnd.scripts.train as st
import BackEnd.scripts.util as su
import BackEnd.scripts.plot as sp

def Run(modelname:str):
    # train은 초반에만 True로 진행 이후는 False
    if modelname == 'Lof':
        params = {
            'n_neighbors': 50,
            'contamination': 0.01,
            'metric': 'manhattan'
        }
    else:
        params = {
            'input_dim': 7,
            'latent_dim': 20,
            'num_embeddings': 64
        }
        
    data, data_exc_key, model = st.train_model(modelname=modelname, params=params, trainable=False)
    
    se_class = su.Util()
    score, y_pred, key_change_indices = se_class.Predict(data=data, data_exc_key=data_exc_key, model_name=modelname, model=model)
    
    sp_class = sp.Plotting(data, score, y_pred, key_change_indices, modelname)
    
    sp_class.draw_lines()
    
    return data, score, y_pred, key_change_indices