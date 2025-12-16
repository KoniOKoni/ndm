#sampling.py
import numpy as np
import time

seed = int(time.time())
_rng = np.random.default_rng(seed)

def sample_params(n_samples : int):
    LogLambda = _rng.uniform(25.0, 25.1, size = n_samples) #Unification scale.
    g0 = _rng.uniform(0.6, 0.7, size = n_samples) #Initial coupling constant unified at Lambda
    Logve = _rng.uniform(12, 15, size = n_samples)
    Logvmu = _rng.uniform(12, 15, size = n_samples)
    Logvtau = _rng.uniform(12, 15, size = n_samples)
    Logg11Re = _rng.uniform(-6, 7, size = n_samples)
    Logg11Im = _rng.uniform(-1e9, -1e8, size = n_samples)
    Logg21Re = _rng.uniform(-6, 7, size = n_samples)
    Logg21Im = _rng.uniform(-6, 7, size = n_samples)
    Logg22Re = _rng.uniform(-6, 7, size = n_samples)
    Logg22Im = _rng.uniform(-1e9, -1e8, size = n_samples)
    Logg31Re = _rng.uniform(-6, 7, size = n_samples)
    Logg31Im = _rng.uniform(-6, 7, size = n_samples)
    Logg32Re = _rng.uniform(-6, 7, size = n_samples)
    Logg32Im = _rng.uniform(-6, 7, size = n_samples)
    Logg33Re = _rng.uniform(-6, 7, size = n_samples)
    Logg33Im = _rng.uniform(-1e9, -1e8, size = n_samples)

    sample_grid = np.stack([LogLambda, g0, Logve, Logvmu, Logvtau,
                            Logg11Re, Logg11Im,
                            Logg21Re, Logg21Im, Logg22Re, Logg22Im,
                            Logg31Re, Logg31Im, Logg32Re, Logg32Im, Logg33Re, Logg33Im], axis=-1)
    idx = {
    "LogLambda": 0,
    "g0": 1,
    "Logve": 2,
    "Logvmu": 3,
    "Logvtau": 4,
    "Logg11Re": 5,
    "Logg11Im": 6,
    "Logg21Re": 7,
    "Logg21Im": 8,
    "Logg22Re": 9,
    "Logg22Im": 10,
    "Logg31Re": 11,
    "Logg31Im": 12,
    "Logg32Re": 13,
    "Logg32Im": 14,
    "Logg33Re": 15,
    "Logg33Im": 16
    }

    return (sample_grid, idx)

