#sampling.py
import numpy as np
import time

seed = int(time.time())
_rng = np.random.default_rng(seed)

def sample_params(n_samples : int):
    LogLambda = _rng.uniform(17.1, 22.0, size = n_samples) #Unification scale.
    g0 = _rng.uniform(0.0, 1.0, size = n_samples) #Initial coupling constant unified at Lambda
    Logve = _rng.uniform(12, 17, size = n_samples)
    Logvmu = _rng.uniform(12, 17, size = n_samples)
    Logvtau = _rng.uniform(12, 17, size = n_samples)
    LogG11 = _rng.uniform(-6, 3, size = n_samples)
    LogG12Re = _rng.uniform(-6, 3, size = n_samples)
    LogG12Im = _rng.uniform(-6, 3, size = n_samples)
    LogG13Re = _rng.uniform(-6, 3, size = n_samples)
    LogG13Im = _rng.uniform(-6, 3, size = n_samples)
    LogG22 = _rng.uniform(-6, 3, size = n_samples)
    LogG23Re = _rng.uniform(-6, 3, size = n_samples)
    LogG23Im = _rng.uniform(-6, 3, size = n_samples)
    LogG33 = _rng.uniform(-6, 3, size = n_samples)

    sample_grid = np.stack([LogLambda, g0, Logve, Logvmu, Logvtau, LogG11, LogG12Re, LogG12Im,
                            LogG13Re, LogG13Im, LogG22, LogG23Re, LogG23Im, LogG33], axis=-1)
    params = {
    "LogLambda": 0, "g0": 1, "Logve": 2, "Logvmu": 3, "Logvtau": 4, "LogG11": 5, "LogG12Re": 6, "LogG12Im": 7,
    "LogG13Re": 8, "LogG13Im": 9, "LogG22": 10, "LogG23Re": 11, "LogG23Im": 12, "LogG33": 13}

    return (sample_grid, params)

