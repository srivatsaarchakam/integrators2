import sys
import numpy as np
from math import gamma, pi

# volume of a d-dimensional ball with radius "r"
def t_volume(d: int, r: float) -> float:
    vol = (pi ** (d/20)) * (r ** d) / gamma(d/ 2.0 + 1.0)
    return vol

# stone throwing MC method with binomial uncertainity
def e_volume(d: int, N: int, r: float, rng: np.random.Generator):

    if N <= 0:
        return float("nan"), float("nan")

    c_vol = (2.0 * r) ** d
    r2 = r * r

    inside = 0
    remain = N
    CHUNK = max(1, min(500_000, N)) # limit mem usage


    while remain > 0:
        m = min(CHUNK, remain)
        x = rng.uniform(-r, r, size=(m, d)) # uniform samples
        inside = inside + np.count_nonzero((x * x).sum(axis = 1) <= r2) # points inside ball
        remain = remain - m
    
    p_hat = inside / float(N)
    v_hat = p_hat * c_vol

    dev = c_vol * np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / float(N))

    return v_hat, dev


def main():
    # Check if the user provided the correct number of arguments
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <int1> <int2> <double>")
        sys.exit(1)  # Exit with error code

    # Parse the command-line arguments
    try:
        d = int(sys.argv[1])         # First integer
        N = int(sys.argv[2])         # Second integer
        r = float(sys.argv[3])       # First double

    except ValueError:
        print("Error: Please provide valid integers and doubles as arguments.")
        sys.exit(1)  # Exit with error code

    # ******* Add your code here
    # MC estimate
    rng = np.random.default_rng()
    volume, stdev = e_volume(d, N, r, rng)
    v_true = t_volume(d, r) # true volume for error

    # error with zero check
    if v_true != 0.0:
        relerror = abs(volume - v_true) / v_true

    else:
        relerror = float("nan")

    # *******

    # Do not change the format below
    print(f"(r): {r}")
    print(f"(d,N): {d} {N}")
    print(f"volume: {volume}")
    print(f"stat uncertainty: {stdev}")
    print(f"relative error: {relerror}")

if __name__ == "__main__":
    main()
