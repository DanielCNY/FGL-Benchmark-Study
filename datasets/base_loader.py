# datasets/base_loader.py
import abc
from typing import Dict, Any, Tuple
import torch

class BaseGraphLoader(abc.ABC):
    """Abstract base class for all federated graph dataset loaders."""
    
    @abc.abstractmethod
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Loads and partitions the graph dataset.
        
        Returns:
            A tuple of (client_datasets, global_test_set).
            - client_datasets: Dict[client_id, client_data_dict]
            - global_test_set: Dict with full graph and test mask
        """
        pass
    
    @abc.abstractmethod
    def get_feature_dim(self) -> int:
        """Return the input feature dimension of the dataset."""
        pass
    
    @abc.abstractmethod
    def get_num_classes(self) -> int:
        """Return the number of output classes."""
        pass