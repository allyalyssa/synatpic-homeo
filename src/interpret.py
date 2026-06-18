import torch
import matplotlib.pyplot as plt
import shap

from config import DEVICE, PATHS
from model import DissipationLSTM

def visualize_attention(model, X_sample):
    model.eval()
    with torch.no_grad():
        X_sample = X_sample.to(DEVICE)
        output = model(X_sample)
    
    explainer = shap.DeepExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 6))
    plt.plot(shap_values[0].mean(axis=1))
    plt.xlabel('Time Step')
    plt.ylabel('SHAP Value')
    plt.title('Attention Weights Over Time')
    plt.savefig(f'{PATHS["outputs"]}/attention_weights.png')
    plt.close()
    
    return shap_values
