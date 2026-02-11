# datasets/base_loader.py
import abc
from typing import Dict, Any, Tuple
import torch

class BaseGraphLoader(abc.ABC):
    
    @abc.abstractmethod
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass
    
    @abc.abstractmethod
    def get_feature_dim(self) -> int:
        pass
    
    @abc.abstractmethod
    def get_num_classes(self) -> int:
        pass