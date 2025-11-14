#main.py
from montecarlo import run_monte_carlo
from sampling import sample_params
from model import model_NDM
from constraints import CLFV_constraints

def main():
    n_samples = 100

    out = run_monte_carlo(
        n_samples = n_samples,
        sample_params = sample_params,
        model = model_NDM,
        constraints = CLFV_constraints
    )

    print(out["masked_results"])
    

if __name__ == "__main__":
    main()