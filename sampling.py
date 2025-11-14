#sampling.py
import numpy as np
import time

seed = int(time.time())
_rng = np.random.default_rng(seed)

def sample_params(n_samples : int):
    LogLambda = _rng.uniform(17.1, 22.0, size = n_samples) #Unification scale.
    g0 = _rng.uniform(0.1, 0.8, size = n_samples) #Initial coupling constant unified at Lambda
    Logve = _rng.uniform(12, 17, size = n_samples)
    Logvmu = _rng.uniform(12, 17, size = n_samples)
    Logvtau = _rng.uniform(12, 17, size = n_samples)
    Logg11Re = _rng.uniform(-6, 3, size = n_samples)
    Logg11Im = _rng.uniform(-6, 3, size = n_samples)
    Logg12Re = _rng.uniform(-6, 3, size = n_samples)
    Logg12Im = _rng.uniform(-6, 3, size = n_samples)
    Logg13Re = _rng.uniform(-6, 3, size = n_samples)
    Logg13Im = _rng.uniform(-6, 3, size = n_samples)
    Logg21Re = _rng.uniform(-6, 3, size = n_samples)
    Logg21Im = _rng.uniform(-6, 3, size = n_samples)
    Logg22Re = _rng.uniform(-6, 3, size = n_samples)
    Logg22Im = _rng.uniform(-6, 3, size = n_samples)
    Logg23Re = _rng.uniform(-6, 3, size = n_samples)
    Logg23Im = _rng.uniform(-6, 3, size = n_samples)
    Logg31Re = _rng.uniform(-6, 3, size = n_samples)
    Logg31Im = _rng.uniform(-6, 3, size = n_samples)
    Logg32Re = _rng.uniform(-6, 3, size = n_samples)
    Logg32Im = _rng.uniform(-6, 3, size = n_samples)
    Logg33Re = _rng.uniform(-6, 3, size = n_samples)
    Logg33Im = _rng.uniform(-6, 3, size = n_samples)

    sample_grid = np.stack([LogLambda, g0, Logve, Logvmu, Logvtau,
                            Logg11Re, Logg11Im, Logg12Re, Logg12Im, Logg13Re, Logg13Im,
                            Logg21Re, Logg21Im, Logg22Re, Logg22Im, Logg23Re, Logg23Im,
                            Logg31Re, Logg31Im, Logg32Re, Logg32Im, Logg33Re, Logg33Im], axis=-1)
    params = {
    "LogLambda": 0,
    "g0": 1,
    "Logve": 2,
    "Logvmu": 3,
    "Logvtau": 4,
    "Logg11Re": 5,
    "Logg11Im": 6,
    "Logg12Re": 7,
    "Logg12Im": 8,
    "Logg13Re": 9,
    "Logg13Im": 10,
    "Logg21Re": 11,
    "Logg21Im": 12,
    "Logg22Re": 13,
    "Logg22Im": 14,
    "Logg23Re": 15,
    "Logg23Im": 16,
    "Logg31Re": 17,
    "Logg31Im": 18,
    "Logg32Re": 19,
    "Logg32Im": 20,
    "Logg33Re": 21,
    "Logg33Im": 22
    }

    return (sample_grid, params)

