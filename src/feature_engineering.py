import numpy as np
import pandas as pd

from config import PATHS

def build_feature_matrix(preprocessed_files):
    features = []
    
    for file in preprocessed_files:
        data = np.load(file)
        subject_id = file.split('/')[-1].replace('_nrem_epochs.npy', '')
        
        mean_amp = np.mean(data)
        std_amp = np.std(data)
        max_amp = np.max(data)
        min_amp = np.min(data)
        
        features.append({
            'subject_id': subject_id,
            'mean_amplitude': mean_amp,
            'std_amplitude': std_amp,
            'max_amplitude': max_amp,
            'min_amplitude': min_amp,
            'n_epochs': data.shape[0]
        })
    
    df = pd.DataFrame(features)
    df.to_csv(f'{PATHS["outputs"]}/feature_matrix.csv', index=False)
    return df
