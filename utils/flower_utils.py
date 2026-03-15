from flwr.server.strategy import FedAvg
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters, FitIns
from flwr.server.client_proxy import ClientProxy
import numpy as np
from typing import Dict, Optional, Tuple, List

def weighted_average(metrics):
    accuracies = [m["accuracy"] for _, m in metrics]
    samples = [m["num_samples"] for _, m in metrics]
    weighted_acc = sum(a * s for a, s in zip(accuracies, samples)) / sum(samples)
    return {"accuracy": weighted_acc}

class SaveModelStrategy(FedAvg):
    def __init__(self, client_config=None, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None
        self.client_config = client_config or {}
        self.client_metrics = {}
        self.global_class_prototypes = None
        self.round_metrics = {}

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            self.client_metrics[cid] = fit_res.metrics

        total_upload = sum(m.get("num_params", 0) for m in self.client_metrics.values())
        total_download = total_upload
        fit_durations = [m.get("fit_duration", 0) for m in self.client_metrics.values()]
        mean_fit_duration = np.mean(fit_durations) if fit_durations else 0.0
        sampled_clients = self.min_fit_clients if hasattr(self, 'min_fit_clients') else len(results) + len(failures)
        participation_rate = len(results) / sampled_clients if sampled_clients > 0 else 0.0
        
        self.round_metrics[server_round] = {
            "total_upload_params": total_upload,
            "total_download_params": total_download,
            "mean_fit_duration": mean_fit_duration,
            "participation_rate": participation_rate,
        }
        
        return aggregated_parameters, aggregated_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []
        for client, fit_ins in instructions:
            new_config = {**fit_ins.config, **self.client_config}
            new_fit_ins = FitIns(fit_ins.parameters, new_config)
            new_instructions.append((client, new_fit_ins))
        return new_instructions