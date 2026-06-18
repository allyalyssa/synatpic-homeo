import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DEVICE, PATHS
from model import DissipationLSTM

def train_model(X_train, y_train, epochs=50, batch_size=32):
    model = DissipationLSTM(input_dim=X_train.shape[2], hidden_dim=64).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    torch.save(model.state_dict(), f'{PATHS["models"]}/dissipation_lstm.pt')
    return model
