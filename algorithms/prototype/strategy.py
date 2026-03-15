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
        self.global_class_prototypes = None
        self.round_metrics = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []
        for client, fit_ins in instructions:
            new_config = {**fit_ins.config, **self.client_config}
            if self.global_class_prototypes is not None:
                serializable = {
                    str(k): v.tolist() for k, v in self.global_class_prototypes.items()
                }
                new_config["global_class_prototypes"] = serializable
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

        all_protos = []
        for client_proxy, fit_res in results:
            protos = fit_res.metrics.get("class_prototypes")
            if protos is not None:
                weight = fit_res.metrics.get("num_samples", 1)
                all_protos.append((protos, weight))

        if all_protos:
            all_classes = set()
            for protos, _ in all_protos:
                all_classes.update(protos.keys())
            global_protos = {}
            total_weight = sum(w for _, w in all_protos)
            for c in all_classes:
                weighted_sum = None
                weight_sum = 0
                for protos, w in all_protos:
                    if c in protos:
                        if weighted_sum is None:
                            weighted_sum = protos[c] * w
                        else:
                            weighted_sum += protos[c] * w
                        weight_sum += w
                if weighted_sum is not None:
                    global_protos[c] = weighted_sum / weight_sum
            self.global_class_prototypes = global_protos
        else:
            self.global_class_prototypes = None

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