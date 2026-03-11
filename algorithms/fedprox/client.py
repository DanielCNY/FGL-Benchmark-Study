import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
from algorithms.base.client import BaseGraphClient 

class FedProxGraphClient(BaseGraphClient):
    def __init__(self, client_data, client_id):
        super().__init__(client_data, client_id)
        self.global_parameters = None

    def set_parameters(self, parameters: List[np.ndarray]):
        super().set_parameters(parameters)
        self.global_parameters = [p.clone().detach().to(self.device)
                                   for p in self.model.parameters()]

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.set_parameters(parameters) 

        x = self.client_data['x'].to(self.device)
        edge_index = self.client_data['edge_index'].to(self.device)
        y = self.client_data['y'].to(self.device)
        train_mask = self.client_data['train_mask'].to(self.device)

        lr = config.get("learning_rate", 0.01)
        local_epochs = config.get("local_epochs", 5)
        mu = config.get("mu", 0.01)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(local_epochs):
            optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])

            if self.global_parameters is not None:
                prox_loss = sum(torch.sum((w - w_global) ** 2)
                                for w, w_global in zip(self.model.parameters(),
                                                    self.global_parameters))
                loss += (mu / 2) * prox_loss

            loss.backward()
            optimizer.step()
            self.last_loss = loss.item()

        num_samples = train_mask.sum().item()
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(x, edge_index)
            pred = out.argmax(dim=1)
            val_mask = self.client_data['val_mask'].to(self.device)
            val_correct = (pred[val_mask] == y[val_mask]).sum().item()
            val_total = val_mask.sum().item()
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        self.model.train()

        metrics = {
            "loss": self.last_loss,
            "accuracy": val_accuracy,
            "client_id": self.client_id,
            "num_samples": num_samples,
            "local_epochs": local_epochs,
            "learning_rate": lr,
            "mu": mu
        }
        return self.get_parameters(config), num_samples, metrics