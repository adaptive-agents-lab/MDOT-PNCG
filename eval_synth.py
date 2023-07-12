
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
    entropy_values = [(max_entropy * 0.1, max_entropy * 0.1),
                      (max_entropy * 0.1, max_entropy * 0.9),
                      (max_entropy * 0.9, max_entropy * 0.9)]

    for h_min, h_max in entropy_values:
        records = []
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
            gamma_sinkhorn = 2 ** 9.
            ms = [5, 7, 9]
            T_s = [math.ceil(gamma_sinkhorn / 2 ** m) for m in ms]
            eps_ = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
            methods = ["ConjugateGradient-S", "Sinkhorn"]  # , "ConjugateGradient-H", "ConjugateGradient-G", "PreconditionedCG", ]

            for eps in eps_:
                # for m in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
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
                    for method in methods:
                        start = time.time()
                        P_0 = th.matmul(mu1.unsqueeze(-1), mu2.unsqueeze(-2))

                        # Choose how to compute descent direction for CG
                        if method.startswith("ConjugateGradient"):
                            if method == "ConjugateGradient-S":
                                descent_dir = "Sinkhorn"
                            elif method == "ConjugateGradient-G":
                                descent_dir = "Gradient"
                            elif method == "ConjugateGradient-H":
                                descent_dir = "DiagonalHessian"
                            elif method == "BFGS":
                                pass
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

                        if T == 4 and args.dim <= 32 and method.startswith("ConjugateGradient") and eps == 1e-10:
                            # Extract the psuedo condition numbers (psc_H and psc_MH) from projection logs for ConjugateGradient-S
                            psc_MH = [log["psc_MH"] for log in logs[method]["proj_logs"]]
                            psc_H = [log["psc_H"] for log in logs[method]["proj_logs"]]
                            # Create a plot with 4 subplots
                            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                            axes = axes.flatten()
                            # Plot the psuedo condition numbers
                            for t in range(T):
                                axes[t].plot(th.stack(psc_MH[t]).cpu(), label="MH")
                                axes[t].plot(th.stack(psc_H[t]).cpu(), label="H")
                                axes[t].set_title("t={}".format(t))
                                axes[t].legend()
                                axes[t].set_yscale("log")
                                axes[t].set_xlabel('Steps $k$')
                                axes[t].set_ylabel('$\lambda_{2n}/\lambda_2$')
                            title = 'Pseudo condition numbers\n'
                            title += '$H_{{min}}={:.2f}$, $H_{{max}}={:.2f}$, ' \
                                     '$\\ \overline{{\gamma}}={:.1f}$'.format(
                                hmin, hmax, gamma_sinkhorn)
                            title += ',\n$\\epsilon={:.2}$'.format(eps)
                            title += ', $n={}$'.format(args.dim)
                            title += ', $m={}$'.format(args.dim_dist)
                            title += ', $\sigma_C={:.2f}$'.format((D - D.min()).std().item())
                            plt.suptitle(title)
                            dir_name = 'figs/gammaT{}_d{}_N{}'.format(gamma_sinkhorn, args.dim_dist, args.dim)
                            if not os.path.exists(dir_name):
                                os.makedirs(dir_name)
                            fname = 'psc_{}_Hmin{:.2f}_Hmax{:.2f}_eps{:.2}.png'.format(method, hmin, hmax, eps)
                            plt.savefig('{}/{}'.format(dir_name, fname))

                        # Add the results to the dataframe
                        records.append({
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

                        if len(logs) == len(methods) and T == 4 and eps == 1e-10:
                            plt.close()
                            # First create a figure with 4 axes
                            fig, axes = plt.subplots(2, T // 2, figsize=(10, 10))
                            axes = axes.flatten()
                            # Plot the KL divergence error over steps k for both methods on the same plot (with 4 axes)
                            for t in range(T):
                                # Subtract the minimum value from all values
                                max_len = 0
                                for method in methods:
                                    vals = logs[method]["proj_logs"][t]["errs"]
                                    vals = th.stack(vals).cpu().numpy()
                                    max_len = max(max_len, len(vals))
                                    axes[t].plot(range(1, len(vals) + 1), vals, label=method)
                                # Add a baseline to the plot of O(k^-1)
                                axes[t].plot(range(1, max_len + 1), vals[0].item() / np.arange(1, max_len + 1),
                                         label='O($k^{{-1}}$)')
                                # Add a baseline to the plot of O(k^-2)
                                axes[t].plot(range(1, max_len + 1), vals[0].item() / np.arange(1, max_len + 1) ** 2,
                                         label='O($k^{{-2}}$)')
                                # Add a horizontal dashed line at epsilon
                                axes[t].axhline(y=eps, color='k', linestyle='--')
                                axes[t].legend()
                                axes[t].set_yscale('log')
                                axes[t].set_xscale('log')
                                axes[t].set_ylim(bottom=eps / 100, top=10 * logs[methods[0]]["proj_logs"][t]["errs"][0].item())
                                axes[t].set_title('t={}'.format(t+1))
                            # Add labels to axes and plot title
                                axes[t].set_xlabel('Steps $k$')
                                axes[t].set_ylabel('$\\rho$')
                            title = 'Bregman divergence (infeasibility) over steps k for both methods\n'
                            # Append H_bar, H_max, and gamma * T to the title
                            title += '$H_{{min}}={:.2f}$, $H_{{max}}={:.2f}$, ' \
                                     '$\\ \overline{{\gamma}}={:.1f}$'.format(
                                hmin, hmax, gamma_sinkhorn)
                            title += ',\n$\\epsilon={:.2}$'.format(eps)
                            title += ', $n={}$'.format(args.dim)
                            title += ', $m={}$'.format(args.dim_dist)
                            title += ', $\sigma_C={:.2f}$'.format((D - D.min()).std().item())

                            fig.suptitle(title)
                            fig.subplots_adjust(wspace=0.6, hspace=0.6)
                            dir_name = 'figs/gammaT{}_d{}_N{}'.format(gamma_sinkhorn, args.dim_dist, args.dim)
                            if not os.path.exists(dir_name):
                                os.makedirs(dir_name)
                            fname = 'convergence_Hmin{:.2f}_Hmax{:.2f}_eps{:.2}.png'.format(hmin, hmax, eps)
                            plt.savefig('{}/{}'.format(dir_name, fname))
                            print("Saved figure to {}/{}".format(dir_name, fname))

        df = pd.DataFrame.from_records(records)
        # Plot the total number of iterations vs. epsilon for each method and each T
        dir_name = 'figs/gammaT{}_d{}_N{}'.format(gamma_sinkhorn, args.dim_dist, args.dim)
        # Make dir if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for method in methods:
            # Get the iteration counts for each epsilon from the dataframe
            color = "green" if method == "Sinkhorn" else "red"
            # Choose triangle, square, or circle markers
            markers = ["o", "s", "^"]
            for T_, marker in zip(T_s, markers):
                iters = []
                for eps in eps_:
                    method_data = df[
                        (df["method"] == method) &
                        (df["eps"] == eps) &
                        (df["gamma * T"] == gamma_sinkhorn) &
                        (df["d"] == args.dim_dist) &
                        (df["N"] == args.dim) &
                        (df["T"] == T_)
                        ]
                    iters.append(np.mean(method_data["k_total"].values))
                ax.plot(eps_, iters, label=method + ', (T={})'.format(T_), marker=marker, color=color)

        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('Total number of iterations vs. $\epsilon$ for each projection method and varying $T$')
        ax.set_xlabel('$\epsilon$')
        ax.set_ylabel('Total number of iterations')
        plt.autoscale(enable=True, axis='both', tight=None)
        title = '$H_{{min}}={:.2f}$, $H_{{max}}={:.2f}$, $\\gamma T={:.1f}$'.format(hmin, hmax, gamma_sinkhorn)
        title += ', \n$n={}$'.format(args.dim)
        title += ', $m={}$'.format(args.dim_dist)
        title += ', $\sigma_C={:.2f}$'.format((D - D.min()).std().item())
        fig.suptitle(title)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fname = 'n_iter_Hmin{:.2f}_Hmax{:.2f}'.format(hmin, hmax)
        plt.savefig('{}/{}.png'.format(dir_name, fname))
        print("Saved figure to {}/{}.png".format(dir_name, fname))

        # Plot wall-clock time vs. epsilon for each method and each T
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for method in methods:
            # Get the iteration counts for each epsilon from the dataframe
            color = "green" if method == "Sinkhorn" else "red"
            # Choose triangle, square, or circle markers
            markers = ["o", "s", "^"]
            for T_, marker in zip(T_s, markers):
                wtimes = []
                for eps in eps_:
                    method_data = df[
                        (df["method"] == method) &
                        (df["eps"] == eps) &
                        (df["gamma * T"] == gamma_sinkhorn) &
                        (df["d"] == args.dim_dist) &
                        (df["N"] == args.dim) &
                        (df["T"] == T_)
                        ]
                    wtimes.append(np.mean(method_data["time"].values))
                ax.plot(eps_, wtimes, label=method + ', (T={})'.format(T_), marker=marker, color=color)

        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('Wall-clock time vs. $\epsilon$ for each projection method and varying $T$')
        ax.set_xlabel('$\epsilon$')
        ax.set_ylabel('Wall-clock time (s)')
        plt.autoscale(enable=True, axis='both', tight=None)
        title = '$H_{{min}}={:.2f}$, $H_{{max}}={:.2f}$, $\\ \overline{{\gamma}}={:.1f}$'.format(hmin, hmax, gamma_sinkhorn)
        title += ', \n$n={}$'.format(args.dim)
        title += ', $m={}$'.format(args.dim_dist)
        title += ', $\sigma_C={:.2f}$'.format((D - D.min()).std().item())
        fig.suptitle(title)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fname = 'wtime_Hmin{:.2f}_Hmax{:.2f}.png'.format(hmin, hmax)
        plt.savefig('{}/{}'.format(dir_name, fname))
        print("Saved figure to {}/{}".format(dir_name, fname))


if __name__ == '__main__':
    main()
