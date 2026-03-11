import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters, FitIns
from flwr.server.client_proxy import ClientProxy
import numpy as np
from typing import Dict, Optional, Tuple, List

class EnvironmentHBDATrategy(fl.server.strategy.FedAvg):
    def __init__(self, env_labels: Dict[str, int], client_config=None, **kwargs):
        super().__init__(**kwargs)
        self.env_labels = env_labels
        self.client_config = client_config or {}
        self.latest_parameters = None
        self.client_metrics = {}

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

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            self.client_metrics[cid] = fit_res.metrics

        env_groups = {}
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            env = self.env_labels.get(cid, 0)
            env_groups.setdefault(env, []).append((client_proxy, fit_res))

        env_params = {}
        env_sizes = {}
        for env, group in env_groups.items():
            params_list = []
            homophily_list = []
            for _, fit_res in group:
                params_list.append(parameters_to_ndarrays(fit_res.parameters))
                homophily_list.append(fit_res.metrics.get("homophily", 0.5))

            weights = np.array(homophily_list) / sum(homophily_list)
            avg_params = [
                np.average([p[layer] for p in params_list], axis=0, weights=weights)
                for layer in range(len(params_list[0]))
            ]
            env_params[env] = avg_params
            env_sizes[env] = len(group)

        total_clients = sum(env_sizes.values())
        global_params = None
        for env, params in env_params.items():
            weight = env_sizes[env] / total_clients
            if global_params is None:
                global_params = [weight * p for p in params]
            else:
                global_params = [gp + weight * p for gp, p in zip(global_params, params)]

        self.latest_parameters = ndarrays_to_parameters(global_params)
        return self.latest_parameters, {}

    def initialize_parameters(self, client_manager):
        return None