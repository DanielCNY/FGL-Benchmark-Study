from datasets.cora import CoraLoader
from models.gcn import GCN

_loader = None
_client_data = None
_test_data = None

def load_partition(partition_id: int, num_partitions: int = 5):
    global _loader, _client_data
    if _loader is None:
        _loader = CoraLoader(num_clients=num_partitions, iid=False)
        _client_data, _ = _loader.load_data()
    client_id = f"client_{partition_id}"
    return _client_data[client_id]

def get_model():
    return GCN(in_channels=1433, hidden_channels=64, out_channels=7)

def get_test_data():
    global _test_data
    if _test_data is None:
        from datasets.cora import CoraLoader
        loader = CoraLoader(num_clients=5)
        _, _test_data = loader.load_data()
    return _test_data