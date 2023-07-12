
import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

from mdot import mdot
from utils.synth_data_generation import sample_distance_matrix, sample_uniform_from_simplex
from utils.algorithmic import *
from utils.generic import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=1, help='Integer index of CUDA device to use or the str "cpu".')
    parser.add_argument('--dim', default=32, type=int, help='Probability vector size (number of points in ambient space).')
    parser.add_argument('--dim_dist', default=32, type=int, help='Dimensionality of the ambient space.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambda_1', default=float('inf'), type=float)
    parser.add_argument('--lambda_2', default=0.02, type=float, help='Sinkhorn entropic-regularization weight.')
    parser.add_argument('--p', default=1., type=float, help='p-Wasserstein distance.')
    parser.add_argument('--work_dir', default='./api_bisim_debug', type=str)
    args = parser.parse_args()

    th.set_default_dtype(th.double)
    print("Using double precision.")

    try:
        os.makedirs(args.work_dir)
    except FileExistsError:
        pass
    set_seed_everywhere(args.seed)
    device = args.device if args.device == 'cpu' else 'cuda:{}'.format(args.device)
    max_entropy = math.floor(math.log2(args.dim))
    entropy_values = [(max_entropy * _,) * 2 for _ in np.arange(0.2, 0.9, 0.1)]

    records = []
    for ent_idx, (h_min, h_max) in enumerate(entropy_values):
        for sample_idx in range(32):
            D = sample_distance_matrix(m=args.dim, n=args.dim, d=args.dim_dist).to(device)
            D = D - D.min()
            D = D / D.max()

            mu1 = sample_uniform_from_simplex(
                args.dim, 1, min_entropy=h_min, max_entropy=h_min + max_entropy * 0.01).to(device)
            h_mu1 = -(mu1 * th.log2(mu1)).sum(-1)

            mu2 = sample_uniform_from_simplex(
                args.dim, 1, min_entropy=h_max, max_entropy=h_max + max_entropy * 0.01).to(mu1.device)
            h_mu2 = -(mu2 * th.log2(mu2)).sum(-1)

            # BEGIN Only to warm up the GPU for wall-clock time measurements
            _, _, _ = mdot(
                th.matmul(mu1.unsqueeze(-1), mu2.unsqueeze(-2)),
                D - D.min(),
                mu1, mu2,
                eps=1e-4, gamma=256, T=1,
                projection_kwargs={
                    "minIter": -1,
                    "maxIter": 1000,
                    "stopping_measure": "bregman",
                },
                warmstart=True)
            # END Only to warm up the GPU for wall-clock time measurements

            hmin = min(h_mu1, h_mu2).item()
            hmax = max(h_mu1, h_mu2).item()
            gamma_sinkhorn = 2 ** 7.
            m = 7
            eps = 1e-5
            methods = ["ConjugateGradient-S", "Sinkhorn", "ConjugateGradient-G"]

            gamma_mdot = 2 ** m
            T = math.ceil(gamma_sinkhorn / gamma_mdot)
            gamma_sinkhorn = gamma_mdot * T
            max_error = hmin / gamma_sinkhorn
            print("\nHmin={:.2f}, \tHmax={:.2f}, \tT={},\tgamma={:.1f},\tmax_err={:.4f}\teps={:.1e}\n".format(
                hmin, hmax, T, gamma_sinkhorn, max_error, eps))

            logs = {}
            for method in methods:
                start = time.time()
                P_0 = th.matmul(mu1.unsqueeze(-1), mu2.unsqueeze(-2))

                # Choose how to compute descent direction for CG
                if method.startswith("ConjugateGradient"):
                    if method == "ConjugateGradient-S":
                        descent_dir = "Sinkhorn"
                    elif method == "ConjugateGradient-G":
                        descent_dir = "Gradient"
                    else:
                        raise ValueError

                P_mdot, k_total, logs[method] = mdot(
                    P_0,
                    D - D.min(),
                    mu1, mu2,
                    eps, gamma_mdot, T,
                    projection_kwargs={
                        "minIter": -1,
                        "maxIter": 2**16 // T,
                        "stopping_measure": "bregman",
                        # Applies only to CG
                        "method": "PR+",
                        "descent_dir": descent_dir if method.startswith("ConjugateGradient") else None,
                    },
                    projection=method.split("-")[0])
                end = time.time()
                print("Method: {}".format(method))
                print("Total time: {:.2f}s".format(end - start))
                print("Total time: {:.2f}s".format(end - start))
                print("Total iterations: {}".format(k_total))
                print("Time per iteration: {:.4f}".format((end-start) / k_total))
                print("Cost: {:.4f}".format(logs[method]["costs"][-1]))
                print("Rounded cost: {:.4f}".format(logs[method]["rounded_cost"]))
                print("Projection error: {:.2e}".format(logs[method]["proj_logs"][-1]["errs"][-1]))
                print("LS func count: {}".format(logs[method]["ls_func_cnt"]))
                print("LS func calls per iteration: {:.2f}".format(logs[method]["ls_func_cnt"] / k_total))
                print()

                # Add the results to the dataframe
                records.append({
                    "H_min": hmin,
                    "H_max": hmax,
                    "ent_idx": ent_idx,
                    "d": args.dim_dist,
                    "N": args.dim,
                    "eps": eps,
                    "rounded_cost": logs[method]["rounded_cost"],
                    "k_total": k_total,
                    "gamma * T": gamma_sinkhorn,
                    "T": T,
                    "time": end - start,
                    "sigma_C": (D - D.min()).std().item(),
                    "method": method,
                    "ls_func_count": logs[method]["ls_func_cnt"],
                    "init_errs": np.array(logs[method]["init_errors"])
                })

    df = pd.DataFrame.from_records(records)
    # Plot the total number of iterations vs. epsilon for each method and each T
    dir_name = 'figs/gammaT{}_d{}_N{}'.format(gamma_sinkhorn, args.dim_dist, args.dim)
    # Make dir if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # In the same plot, show the total number of iterations vs. h_min for each method
    for method in methods:
        method_data = df[df["method"] == method]
        method_data_mean = method_data.groupby("ent_idx").mean()
        method_data_std = method_data.groupby("ent_idx").std()
        n_iters = method_data_mean["k_total"]
        h_mins = method_data_mean["H_min"]
        ax.plot(h_mins / max_entropy, n_iters, label=method)
        ax.fill_between(h_mins / max_entropy, n_iters - method_data_std["k_total"] / math.sqrt(32),
                        n_iters + method_data_std["k_total"] / math.sqrt(32), alpha=0.2)
    ax.legend()
    ax.set_yscale('log')
    ax.set_title('Total number of iterations vs. $H_{\min} ~/~ \log_2 n$ for each projection method')
    ax.set_xlabel('$H_{\min} ~/~ \log n$')
    ax.set_ylabel('Total number of iterations')
    plt.autoscale(enable=True, axis='both', tight=None)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    print()



if __name__ == '__main__':
    main()
