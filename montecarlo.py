#montecarlo.py
import numpy as np

def run_monte_carlo(n_samples, sample_params, model, constraints):
    params, idx = sample_params(n_samples)
    results = model(params, idx)
    mask = constraints(results)

    accepted_params = params[mask]
    accepted_results = {k: v[mask] for k, v in results.items()}

    return {
        "params": params,
        "idx": idx,
        "results": results,
        "mask": mask,
        "accepted_params": accepted_params,
        "accepted_results": accepted_results
        }
