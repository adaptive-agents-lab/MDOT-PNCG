
import argparse
import math

import matplotlib.pyplot as plt

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
    h_min, h_max = max_entropy * 0.9, max_entropy * 0.9

    gamma_sinkhorn = 2 ** 9.
    ms = [1, 3, 5, 7]
    T_s = [math.ceil(gamma_sinkhorn / 2 ** m) for m in ms]
    eps = 1e-10
    method = "ConjugateGradient-S"

    records = {T_: [] for T_ in T_s}
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

        hmin = min(h_mu1, h_mu2).item()
        hmax = max(h_mu1, h_mu2).item()

        for m in ms:
            gamma_mdot = 2 ** m
            # gamma_mdot = 2e-0 / (D - D.min()).max()
            T = math.ceil(gamma_sinkhorn / gamma_mdot)
            gamma_sinkhorn = gamma_mdot * T
            # T = max(1, math.floor(1 * hbar))d
            # print("2 gamma Cinf = {:.2f}".format(2*gamma_sinkhorn * (D - D.min()).max()))
            max_error = hmin / gamma_sinkhorn
            print("\nHmin={:.2f}, \tHmax={:.2f}, \tT={},\tgamma={:.1f},\tmax_err={:.4f}\teps={:.1e}\n".format(
                hmin, hmax, T, gamma_sinkhorn, max_error, eps))

            logs = {}
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
            records[T].append({
                "H_min": hmin,
                "H_max": hmax,
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

    plt.close()
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, T_ in enumerate(T_s):
        init_errs_mean = np.array([r["init_errs"] for r in records[T_]]).mean(0)
        init_errs_std = np.array([r["init_errs"] for r in records[T_]]).std(0)
        ax.plot(np.arange(1, T_ + 1), init_errs_mean, label='T={}'.format(T_))
        ax.fill_between(np.arange(1, T_ + 1), init_errs_mean - init_errs_std,
                        init_errs_mean + init_errs_std, alpha=0.2)
        # Add horizontal line at eps
    ax.axhline(y=eps, color='black', linestyle='--')
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_ylabel('$\\rho_0$')
    ax.set_xlabel('$t$')

    title = '$H_{{min}}={:.2f}$, $H_{{max}}={:.2f}$, \n$\gamma T={:.1f}$'.format(hmin, hmax,
                                                                                             gamma_sinkhorn)
    title += ', $n={}$'.format(args.dim)
    title += ', $m={}$'.format(args.dim_dist)
    title += ', $\sigma_C={:.2f}$'.format((D - D.min()).std().item())
    plt.title(title)
    dir_name = 'figs/gammaT{}_d{}_N{}'.format(gamma_sinkhorn, args.dim_dist, args.dim)
    fname = 'init_error_Hmin{:.2f}_Hmax{:.2f}.png'.format(hmin, hmax)
    plt.savefig('{}/{}'.format(dir_name, fname))
    print("Saved figure to {}/{}".format(dir_name, fname))
    print()


if __name__ == '__main__':
    main()
