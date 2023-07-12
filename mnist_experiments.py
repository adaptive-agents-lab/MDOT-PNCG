
import argparse
import math
import ot
import torch

import matplotlib.pyplot as plt
import pandas as pd

from torchvision import transforms, datasets

from mdot import mdot
from utils.algorithmic import *
from utils.generic import *


def prepare_datasets(device, n_images=1000):
    # Load MNIST dataset
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    # Randomly sample 2000 images from the dataset without replacement
    mnist_train = torch.utils.data.Subset(mnist_train, torch.randperm(len(mnist_train))[:2*n_images])

    # Split the data into two sets of N images each
    mnist_train_1, mnist_train_2 = torch.utils.data.random_split(mnist_train, [n_images, n_images])

    # Construct a 784x2 matrix with pixel locations of a 28 * 28 image
    x = torch.zeros((28 * 28, 2))
    for i in range(28):
        for j in range(28):
            x[i * 28 + j, 0] = i
            x[i * 28 + j, 1] = j
    # Compute pairwise L1 distances between pixel locations.
    D = (x.unsqueeze(1) - x.unsqueeze(0)).abs().sum(-1)

    # Flatten the images into vectors
    mu1 = torch.zeros((n_images, 28 * 28))
    mu2 = torch.zeros((n_images, 28 * 28))
    for i in range(n_images):
        mu1[i, :] = mnist_train_1[i][0].flatten()
        mu2[i, :] = mnist_train_2[i][0].flatten()

    # Add 1e-6 to avoid numerical issues
    mu1 += 1e-6 * torch.rand_like(mu1)
    mu2 += 1e-6 * torch.rand_like(mu1)

    # Normalize the images to be probability vectors
    mu1 /= mu1.sum(dim=1, keepdim=True)
    mu2 /= mu2.sum(dim=1, keepdim=True)

    D /= D.max()

    return mu1.to(device), mu2.to(device), D.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=1, help='Integer index of CUDA device to use or the str "cpu".')
    parser.add_argument('--n_images', default=512, type=int, help='Number of images to use from the MNIST dataset.')
    args = parser.parse_args()
    th.set_default_dtype(th.double)
    print("Using double precision.")
    device = args.device if args.device == 'cpu' else 'cuda:{}'.format(args.device)
    mu1s, mu2s, D = prepare_datasets(device, n_images=args.n_images)

    hmax = math.log2(mu1s.size(-1))
    # Compute entropies of the distributions
    h_mu1 = -(mu1s * th.log2(mu1s)).sum(-1)
    h_mu2 = -(mu2s * th.log2(mu2s)).sum(-1)

    # Plot the histogram of entropy values for the concatenation of h_mu1 and h_mu2
    plt.hist(torch.cat((h_mu1, h_mu2)).cpu().numpy(), bins=25)
    # Add vertical dashed line at hmax
    plt.axvline(x=hmax, color='k', linestyle='--', linewidth=0.75)
    # Mark the dashed line on the left side as log2(784) at half the image height
    plt.text(hmax - 0.9, 0.7 * plt.ylim()[1], r'$\log_2(784)$', fontsize=12, horizontalalignment='center')

    plt.xlabel('Entropy')
    plt.xlim(0., 1.1 * hmax)
    plt.ylabel('Frequency')
    plt.title('Histogram of entropy values for the MNIST dataset')
    plt.show()

    # BEGIN Only to warm up the GPU for wall-clock time measurements
    _, _, _ = mdot(
        th.matmul(mu1s[0].unsqueeze(0).unsqueeze(-1), mu2s[0].unsqueeze(0).unsqueeze(-2)),
        D - D.min(),
        mu1s[0].unsqueeze(0), mu2s[0].unsqueeze(0),
        eps=1e-4, gamma=256, T=1,
        projection_kwargs={
            "minIter": -1,
            "maxIter": 1000,
            "stopping_measure": "bregman",
        },
        warmstart=True)
    # END Only to warm up the GPU for wall-clock time measurements

    gamma_bars = [1, 32, 128, 256, 512, 4096]
    gamma_max = 256.
    T_s = [max(math.floor(gamma_bar / gamma_max), 1) for gamma_bar in gamma_bars]
    eps = 1e-11
    methods = ["ConjugateGradient-S", "Sinkhorn"]
    records = []

    for j in range(args.n_images):
        print("Problem {}".format(j))
        mu1 = mu1s[j, :].unsqueeze(0)
        mu2 = mu2s[j, :].unsqueeze(0)

        # Compute the ground truth transport plan
        exact_sol = ot.emd(mu1.cpu().numpy().squeeze(), mu2.cpu().numpy().squeeze(), D.cpu().numpy(), numItermax=1e8)
        exact_cost = (exact_sol * D.cpu().numpy()).sum()

        for i, gamma_bar in enumerate(gamma_bars):
            T = T_s[i]
            gamma_mdot = gamma_bar / T
            print("\nT={},\tgamma={:.1f},\teps={:.1e}\n".format(
                T, gamma_bar, eps))

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
                        "maxIter": 2 ** 24 // T,
                        "stopping_measure": "bregman",
                        # Applies only to CG
                        # "method": "PR+",
                        "method": "PCG",
                        "descent_dir": descent_dir if method.startswith("ConjugateGradient") else None,
                    },
                    projection=method.split("-")[0],
                    warmstart=True)
                end = time.time()
                print("Method: {}".format(method))
                print("Total time: {:.2f}s".format(end - start))
                print("Total time: {:.2f}s".format(end - start))
                print("Total iterations: {}".format(k_total))
                print("Time per iteration: {:.4f}".format((end - start) / k_total))
                print("Cost: {:.4f}".format(logs[method]["costs"][-1]))
                print("Rounded cost: {:.4f}".format(logs[method]["rounded_cost"]))
                print("Relative error: {:.4e}".format((logs[method]["rounded_cost"] - exact_cost) / exact_cost))
                print("Projection error: {:.4e}".format(logs[method]["proj_logs"][-1]["errs"][-1]))
                print("LS func count: {}".format(logs[method]["ls_func_cnt"]))
                print("LS func calls per iteration: {:.2f}".format(logs[method]["ls_func_cnt"] / k_total))
                print()

                # Add the results to the dataframe
                records.append({
                    "idx": j,
                    "eps": eps,
                    "rounded_cost": logs[method]["rounded_cost"],
                    "error": (logs[method]["rounded_cost"] - exact_cost),
                    "relative_error": (logs[method]["rounded_cost"] - exact_cost) / exact_cost,
                    "k_total": k_total,
                    "gamma * T": gamma_bar,
                    "T": T,
                    "time": end - start,
                    "sigma_C": (D - D.min()).std().item(),
                    "method": method,
                    "ls_func_count": logs[method]["ls_func_cnt"],
                })

            print()

    colors = {"Sinkhorn": "red", "ConjugateGradient-S": "green"}
    df = pd.DataFrame.from_records(records)

    plt.close()
    for method in methods:
        for g in gamma_bars:
            # extract relative error and number of iterations for the given method and gamma
            x = df[(df.method == method) & (df["gamma * T"] == g)].relative_error
            y = df[(df.method == method) & (df["gamma * T"] == g)].k_total
            mean_x = x.mean()
            mean_y = y.mean()
            # Compute confidence intervals
            # Calculate the 95th and 5th percentiles
            lower_x = np.percentile(x, 5)
            upper_x = np.percentile(x, 95)
            lower_y = np.percentile(y, 5)
            upper_y = np.percentile(y, 95)
            # Outliers
            outliers_x = x[(x < lower_x) | (x > upper_x)]
            outliers_y = y[(y < lower_y) | (y > upper_y)]
            plt.hlines(mean_y, lower_x, upper_x)
            plt.vlines(mean_x, lower_y, upper_y)
            # Confidence interval bars
            plt.errorbar(mean_x, mean_y,
                         xerr=[[mean_x - lower_x], [upper_x - mean_x]],
                         yerr=[[mean_y - lower_y], [upper_y - mean_y]],
                         fmt='s', capsize=5, color=colors[method])
            # Outliers
            plt.scatter(outliers_x, np.repeat(mean_y, len(outliers_x)), color=colors[method], marker='x', s=5,
                        alpha=0.1)
            plt.scatter(np.repeat(mean_x, len(outliers_y)), outliers_y, color=colors[method], marker='x', s=10,
                        alpha=0.1)
    # Add legends corresponding to the color given by each method
    plt.legend([plt.Line2D([0], [0], color=colors[method], lw=4) for method in methods], methods)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Number of iterations")
    plt.xlabel("Relative error")
    plt.title("Relative error vs. number of iterations (MNIST)")
    plt.show()

    plt.close()
    for method in methods:
        for g in gamma_bars:
            # extract relative error and number of iterations for the given method and gamma
            x = df[(df.method == method) & (df["gamma * T"] == g)].relative_error
            y = df[(df.method == method) & (df["gamma * T"] == g)].time
            mean_x = x.mean()
            mean_y = y.mean()
            # Compute confidence intervals
            # Calculate the 95th and 5th percentiles
            lower_x = np.percentile(x, 5)
            upper_x = np.percentile(x, 95)
            lower_y = np.percentile(y, 5)
            upper_y = np.percentile(y, 95)
            # Outliers
            outliers_x = x[(x < lower_x) | (x > upper_x)]
            outliers_y = y[(y < lower_y) | (y > upper_y)]
            plt.hlines(mean_y, lower_x, upper_x)
            plt.vlines(mean_x, lower_y, upper_y)
            # Confidence interval bars
            plt.errorbar(mean_x, mean_y,
                         xerr=[[mean_x - lower_x], [upper_x - mean_x]],
                         yerr=[[mean_y - lower_y], [upper_y - mean_y]],
                         fmt='s', capsize=5, color=colors[method])
            # Outliers
            plt.scatter(outliers_x, np.repeat(mean_y, len(outliers_x)), color=colors[method], marker='x', s=5,
                        alpha=0.1)
            plt.scatter(np.repeat(mean_x, len(outliers_y)), outliers_y, color=colors[method], marker='x', s=10,
                        alpha=0.1)
    # Add legends corresponding to the color given by each method
    plt.legend([plt.Line2D([0], [0], color=colors[method], lw=4) for method in methods], methods)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Wall-clock time (s)")
    plt.xlabel("Relative error")
    plt.title("Relative error vs. wall-clock time (MNIST)")
    plt.show()
    print()




if __name__ == '__main__':
    main()
