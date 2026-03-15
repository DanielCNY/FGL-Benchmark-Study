import torch
import flwr as fl
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from models.gcn import GCN
HIDDEN_DIM = 64

class BaseGraphClient(fl.client.NumPyClient):

    def __init__(self, client_data: Dict[str, Any], client_id: str, model=None):
        self.client_data = client_data
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model is None:
            feature_dim = client_data['x'].shape[1]
            num_classes = len(torch.unique(client_data['y']))
            self.model = GCN(in_channels=feature_dim, hidden_channels=HIDDEN_DIM, out_channels=num_classes)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        self.optimizer = None
        self.last_loss = None
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        if self.model is None:
            raise ValueError("Model not initialized. Call set_parameters() first.")

        return [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = {
            k: torch.tensor(v) 
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        
        self.model.load_state_dict(state_dict, strict=True)
    
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()
        
        x = self.client_data['x'].to(self.device)
        edge_index = self.client_data['edge_index'].to(self.device)
        y = self.client_data['y'].to(self.device)
        val_mask = self.client_data['val_mask'].to(self.device)
        
        with torch.no_grad():
            out = self.model(x, edge_index)
            
            loss = torch.nn.functional.nll_loss(out[val_mask], y[val_mask])
            
            pred = out.argmax(dim=1)
            correct = (pred[val_mask] == y[val_mask]).sum()
            accuracy = correct.item() / val_mask.sum().item()
        
        metrics = {
            "accuracy": accuracy,
            "loss": loss.item(),
            "client_id": self.client_id,
            "num_samples": len(val_mask)
        }
        
        return float(loss), len(val_mask), metrics
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement fit() method!")
    
    def get_num_train_samples(self) -> int:
        return self.client_data['train_mask'].sum().item()
    
    def get_client_info(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "num_nodes": len(self.client_data['node_ids']),
            "num_train": self.get_num_train_samples(),
            "num_val": self.client_data['val_mask'].sum().item()
        }
    
    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())
