import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters, FitIns
from flwr.server.client_proxy import ClientProxy
import numpy as np
from typing import Dict, Optional, Tuple, List

class FedNovaStrategy(fl.server.strategy.FedAvg):
    def __init__(self, eta_global: float = 1.0,
                 client_config: Optional[Dict] = None,
                 initial_parameters: Optional[Parameters] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eta_global = eta_global
        self.client_config = client_config or {}
        self.current_global_weights = None
        self.latest_parameters = initial_parameters
        self.client_metrics = {}
        self.global_class_prototypes = None
        self.round_metrics = {}

        if initial_parameters is not None:
            self.current_global_weights = parameters_to_ndarrays(initial_parameters)

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

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:
        if not results:
            return None, {}

        if self.current_global_weights is None:
            _, first_fit_res = results[0]
            self.current_global_weights = parameters_to_ndarrays(first_fit_res.parameters)

        total_weight = 0.0
        weighted_sum = None

        for client_proxy, fit_res in results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            tau = fit_res.metrics["tau"]
            num_samples = fit_res.metrics["num_samples"]

            delta = [(cw - gw) / tau for cw, gw in zip(client_weights, self.current_global_weights)]

            weight = num_samples * tau
            total_weight += weight

            if weighted_sum is None:
                weighted_sum = [w * delta_layer for w, delta_layer in zip([weight]*len(delta), delta)]
            else:
                weighted_sum = [ws + w * delta_layer for ws, w, delta_layer in zip(weighted_sum, [weight]*len(delta), delta)]

        total_upload = sum(m.get("num_params", 0) for m in self.client_metrics.values())
        total_download = total_upload
        fit_durations = [m.get("fit_duration", 0) for m in self.client_metrics.values()]
        mean_fit_duration = np.mean(fit_durations) if fit_durations else 0.0
        sampled_clients = self.min_fit_clients if hasattr(self, 'min_fit_clients') else len(results) + len(failures)
        participation_rate = len(results) / sampled_clients if sampled_clients > 0 else 0.0
        
        self.round_metrics[server_round] = {
            "participation_rate": participation_rate,
            "total_upload_params": total_upload,
            "total_download_params": total_download,
            "mean_fit_duration": mean_fit_duration,
        }

        avg_delta = [ws / total_weight for ws in weighted_sum]

        new_weights = [gw + self.eta_global * ad for gw, ad in zip(self.current_global_weights, avg_delta)]

        self.current_global_weights = new_weights
        self.latest_parameters = ndarrays_to_parameters(new_weights)

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            self.client_metrics[cid] = fit_res.metrics

        return self.latest_parameters, {}

    def initialize_parameters(self, client_manager):
        if self.current_global_weights is not None:
            return ndarrays_to_parameters(self.current_global_weights)
        return None