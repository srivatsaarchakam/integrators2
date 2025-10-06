import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from math import sqrt
from 3630start import t_volume, e_volume

def one_dimension_errors(d, r, Ns, trials, base_rng):
    v_true = t_volume(d, r)
    inv_sqrtN = []
    mean_err = []
    sem_err = []  # standard error of the mean 

    for N in Ns:
        errs = []
        for _ in range(trials):
            child = np.random.default_rng(base_rng.integers(0, 2 ** 63 - 1))
            vhat, _ = e_volume(d, N, r, child)
            errs.append(abs(vhat - v_true) / v_true)

        errs = np.asarray(errs, dtype=float) 
        inv_sqrtN.append(1.0 / np.sqrt(N))

        mean_err.append(errs.mean())
        sem_err.append(errs.std(ddof = 1) / np.sqrt(len(errs)))  # error bars

    return np.asarray(inv_sqrtN), np.asarray(mean_err), np.asarray(sem_err)

def main():
    p = argparse.ArgumentParser(description = "MC convergence d-D volume")
    p.add_argument("--dims", type = str, default = "3,5,10")
    
    p.add_argument("--r", type = float, default = 1.0, help="Sphere Radius (default = 1.0)")
    p.add_argument("--kmin", type = int, default = 1)
    
    p.add_argument("--kmax", type = int, default = 16)
    p.add_argument("--trials", type = int, default = 16, help="Trials N (default = 1)")

    p.add_argument("--seed", type = int, default = 12345)
    p.add_argument("--out", type = str, default = "convergence.png")
    
    args = p.parse_args()

    dims = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    Ns = [2 ** k for k in range(args.kmin, args.kmax + 1)]
    rng = default_rng(args.seed)

    plt.figure()
    for d in dims:
        x, y, yerr = one_dimension_errors(d, args.r, Ns, args.trials, rng)
        plt.errorbar(x, y, yerr = yerr, fmt = "o", capsize = 3, label = f"d = {d}")

    plt.xlabel("1 / sqrt(N)")
    plt.ylabel("Fractional Error")
    title_dims = ",".join(str(d) for d in dims)
    plt.title(f"Monte Carlo (r={args.r}); {args.trials} trials N; dims = {title_dims}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi = 200)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
