#montecarlo.py
import numpy as np

def run_monte_carlo(n_samples, sample_params, model, constraints):
    samples, params = sample_params(n_samples)
    physical_quantities = model(samples, params)
    mask = constraints(physical_quantities)

    accepted_params = samples[mask]

    return {
        'params' : params,
        'results' : physical_quantities,
        'masked_results' : accepted_params
        }
