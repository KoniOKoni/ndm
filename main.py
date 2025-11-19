#main.py
import getdist.plots
from montecarlo import run_monte_carlo
from sampling import sample_params
from model import model_NDM
from constraints import CLFV_constraints
import numpy as np
import matplotlib.pyplot as plt
import getdist


def main():
    n_samples = 1000

    out = run_monte_carlo(
        n_samples = n_samples,
        sample_params = sample_params,
        model = model_NDM,
        constraints = CLFV_constraints
    )

    params = out["params"]
    idx = out["idx"]
    accepted_params = out["accepted_params"]

    n_params = len(idx)
    names = [None] * n_params
    for name, i in idx.items():
        names[i] = name

    label_map = {"LogLambda" : r'\log\Lambda',
                 'g0' : r'g_0',
                 'Logve' : r'\log v_e',
                 'Logvmu' : r'\log v_\mu',
                 'Logvtau' : r'\log v_tau'}
    
    labels = [label_map.get(n,n) for n in names]

    samples = getdist.MCSamples(samples=accepted_params, names = names, labels = labels)

    chosen = ['LogLambda', 'Logve', 'Logvmu', 'Logvtau']
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot(samples,params=chosen ,filled=True)
    plt.show()
if __name__ == "__main__":
    main()