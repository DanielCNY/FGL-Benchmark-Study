import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np

from algorithms.base.client import BaseGraphClient

class FedAvgGraphClient(BaseGraphClient):
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()
        
        x = self.client_data['x'].to(self.device)
        edge_index = self.client_data['edge_index'].to(self.device)
        y = self.client_data['y'].to(self.device)
        train_mask = self.client_data['train_mask'].to(self.device)
        
        learning_rate = config.get("learning_rate", 0.01)
        local_epochs = config.get("local_epochs", 5)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(local_epochs):
            optimizer.zero_grad()
            
            out = self.model(x, edge_index)
            
            loss = F.nll_loss(out[train_mask], y[train_mask])
            
            loss.backward()
            
            optimizer.step()
            
            self.last_loss = loss.item()
            
            if epoch % 2 == 0:
                print(f"[Client {self.client_id}] Epoch {epoch}, Loss: {loss.item():.4f}")
        
        num_samples = train_mask.sum().item()
        metrics = {
            "loss": self.last_loss,
            "client_id": self.client_id,
            "num_samples": num_samples,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate
        }
        
        updated_parameters = self.get_parameters(config)
        return updated_parameters, num_samples, metrics
    
    def __str__(self) -> str:
        info = self.get_client_info()
        return f"FedAvgClient(id={info['client_id']}, nodes={info['num_nodes']}, train={info['num_train']})"
