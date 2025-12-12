# mcmc_emcee.py
import numpy as np
import emcee

from model import model_NDM
from constraints import CLFV_constraints

# 전체 파라미터 이름 (sampling.py의 stack 순서와 동일하게)
ALL_NAMES = [
    "LogLambda", "g0", "Logve", "Logvmu", "Logvtau",
    "Logg11Re", "Logg11Im",
    "Logg21Re", "Logg21Im", "Logg22Re", "Logg22Im",
    "Logg31Re", "Logg31Im", "Logg32Re", "Logg32Im", "Logg33Re", "Logg33Im"
]

IDX = {name: i for i, name in enumerate(ALL_NAMES)}

# 고정시켜둔 imaginary parts
FIXED_IM_PARAMS = {
    "Logg11Im": 0,
    "Logg22Im": 0,
    "Logg33Im": 0,
    "LogLambda": 25
}

# 실제로 MCMC에서 도는 파라미터 이름
VARYING_NAMES = [name for name in ALL_NAMES if name not in FIXED_IM_PARAMS]

# 기존 sampling.py에서 쓰던 bounds 재현
BOUNDS = {
    #"LogLambda": (17.1, 25.0),
    "g0": (0.6, 0.7),
    "Logve": (12.0, 15.0),
    "Logvmu": (12.0, 15.0),
    "Logvtau": (12.0, 15.0),

    "Logg11Re": (-6.0, 7.0),
    # "Logg11Im" 고정

    "Logg21Re": (-6.0, 7.0),
    "Logg21Im": (-6.0, 7.0),
    "Logg22Re": (-6.0, 7.0),
    # "Logg22Im" 고정

    "Logg31Re": (-6.0, 7.0),
    "Logg31Im": (-6.0, 7.0),
    "Logg32Re": (-6.0, 7.0),
    "Logg32Im": (-6.0, 7.0),
    "Logg33Re": (-6.0, 7.0),
    # "Logg33Im" 고정
}


NDIM = len(VARYING_NAMES)


def _build_full_param_vector(theta):
    """
    theta: shape (NDIM,) for VARYING_NAMES 순서
    반환: shape (17,) for ALL_NAMES 순서
    """
    full = np.empty(len(ALL_NAMES), dtype=float)

    # varying parameters 채우기
    for i, name in enumerate(VARYING_NAMES):
        full[IDX[name]] = theta[i]

    # fixed imaginary parts 채우기
    for name, val in FIXED_IM_PARAMS.items():
        full[IDX[name]] = val

    return full


def log_prior(theta):
    """
    Uniform prior in the same ranges as sampling.py.
    범위 밖이면 -inf, 범위 안이면 0.
    """
    for i, name in enumerate(VARYING_NAMES):
        lo, hi = BOUNDS[name]
        x = theta[i]
        if (x < lo) or (x > hi):
            return -np.inf
    return 0.0


def log_likelihood(theta):
    """
    constraints를 top-hat likelihood로 사용.
    constraints를 만족하면 0, 아니면 -inf.
    """
    full_vec = _build_full_param_vector(theta)              # (17,)
    params   = full_vec[np.newaxis, :]                      # (1,17)

    # model_NDM는 여러 샘플을 한 번에 처리하게 돼 있으므로 그대로 사용
    results = model_NDM(params, IDX)

    # CLFV_constraints도 vectorized라 mask shape가 (N,) 형태
    const = CLFV_constraints(results)  # shape (1,)
    if not const[0][0]:
        return -np.inf   # allowed region: constant likelihood
    return const[1]


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def run_emcee(
    nwalkers=32,
    nsteps=10000,
    burnin=2000,
    random_seed=None
):
    """
    emcee를 사용해 허용 영역을 샘플링.
    반환:
        flat_samples: shape (Nsamples, NDIM) for VARYING_NAMES
        log_prob: shape (Nsamples,)
    """
    rng = np.random.default_rng(random_seed)

    # 초기 walker 위치: prior 안에서 랜덤
    p0 = []
    for _ in range(nwalkers):
        pos = np.empty(NDIM, dtype=float)
        for i, name in enumerate(VARYING_NAMES):
            lo, hi = BOUNDS[name]
            pos[i] = rng.uniform(lo, hi)
        p0.append(pos)
    p0 = np.array(p0)

    sampler = emcee.EnsembleSampler(nwalkers, NDIM, log_probability)

    sampler.run_mcmc(p0, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=burnin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burnin, flat=True)

    return flat_samples, flat_log_prob