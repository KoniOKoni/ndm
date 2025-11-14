#main.py
from montecarlo import run_monte_carlo
from sampling import sample_params

def main():
    n_samples = 100

    out = run_monte_carlo(
        n_samples = n_samples,
        sample_params = sample_params
    )

if __name__ == "__main__":
    main()