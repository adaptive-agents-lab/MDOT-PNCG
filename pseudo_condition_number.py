
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
    ent_scales = np.arange(0.2, 0.9, 0.2)
    entropy_values = [(max_entropy * _, ) * 2 for _ in ent_scales]
    # Generate a color for each entropy scale
    colors = [plt.cm.viridis(_) for _ in np.linspace(0, 1, len(ent_scales))]
    labels = ["H={:.2f} $\log_2 n$".format(_) for _ in ent_scales]

    gamma_sinkhorn = 2 ** 5.
    T = 1
    eps = -1
    method = "ConjugateGradient-S"
    gamma_mdot = gamma_sinkhorn / T

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, (h_min, h_max) in enumerate(entropy_values):
        ratios = []
        for sample_idx in range(32):
            mu1 = sample_uniform_from_simplex(
                args.dim, 1, min_entropy=h_min, max_entropy=h_min + max_entropy * 0.01).to(device)
            h_mu1 = -(mu1 * th.log2(mu1)).sum(-1)

            mu2 = sample_uniform_from_simplex(
                args.dim, 1, min_entropy=h_max, max_entropy=h_max + max_entropy * 0.01).to(mu1.device)
            h_mu2 = -(mu2 * th.log2(mu2)).sum(-1)

            D = sample_distance_matrix(m=args.dim, n=args.dim, d=args.dim_dist).to(device)
            D = D - D.min()
            D = D / D.max()

            # BEGIN Only to warm up the GPU for wall-clock time measurements
            _, _, _ = mdot(
                th.matmul(mu1.unsqueeze(-1), mu2.unsqueeze(-2)),
                D - D.min(),
                mu1, mu2,
                eps=1e-4, gamma=256, T=1,
                projection_kwargs={
                    "minIter": 30,
                    "maxIter": 30,
                    "stopping_measure": "bregman",
                },
                warmstart=True)
            # END Only to warm up the GPU for wall-clock time measurements

            hmin = min(h_mu1, h_mu2).item()
            hmax = max(h_mu1, h_mu2).item()

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
                    "minIter": 50,
                    "maxIter": 50,
                    "stopping_measure": "bregman",
                    # Applies only to CG
                    "method": "PR+",
                    "descent_dir": descent_dir if method.startswith("ConjugateGradient") else None,
                    "compute_psc": True
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

            # Extract the psuedo condition numbers (psc_H and psc_MH) from projection logs for ConjugateGradient-S
            psc_MH = [log["psc_MH"] for log in logs[method]["proj_logs"]][0]
            psc_H = [log["psc_H"] for log in logs[method]["proj_logs"]][0]

            ratios.append(th.stack(psc_H).cpu()/ th.stack(psc_MH).cpu())

        # Plot the psuedo condition numbers
        ax.plot(th.stack(ratios).mean(0), label="{}".format(labels[i]), color=colors[i])

    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel('Steps $k$')
    title = 'Ratio of pseudo condition numbers for $\\nabla^2 g$ and $M \\nabla^2 g$\n'
    title += '$\\ \overline{{\gamma}}={:.1f}, T=1$'.format(gamma_sinkhorn)
    title += ', $n={}$'.format(args.dim)
    title += ', $m={}$'.format(args.dim_dist)
    title += ', $\sigma_C={:.2f}$'.format((D - D.min()).std().item())
    plt.title(title)

    dir_name = 'figs/gammaT{}_d{}_N{}'.format(gamma_sinkhorn, args.dim_dist, args.dim)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.show()


if __name__ == '__main__':
    main()
