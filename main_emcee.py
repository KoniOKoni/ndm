# main_emcee.py
import numpy as np
import matplotlib.pyplot as plt
import getdist
import getdist.plots

from mcmc_emcee import run_emcee, VARYING_NAMES

def main():
    # MCMC 실행
    flat_samples, flat_log_prob = run_emcee(
        nwalkers=32,
        nsteps=100000,
        burnin=5000,
        random_seed=42
    )

    names = VARYING_NAMES

    # 라벨: 기존 main.py의 label_map 재사용
    label_map = {
        "LogLambda": r'\log\Lambda',
        'g0': r'g_0',
        'Logve': r'\log v_e',
        'Logvmu': r'\log v_\mu',
        'Logvtau': r'\log v_\tau'
    }

    labels = [label_map.get(n, n) for n in names]

    # getdist MCSamples
    samples = getdist.MCSamples(samples=flat_samples, names=names)
    samples.saveAsText('ndm.txt')

    # 보고 싶은 파라미터만 선택 (이름이 VARYING_NAMES에 들어 있는 것들)
    chosen = [p for p in names]

    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot(samples, params=chosen, filled=True)
    plt.show()


if __name__ == "__main__":
    main()