
import torch as th
import torch.nn.functional as F


def sample_uniform_from_simplex(N, n_samples, min_entropy=None, max_entropy=None, minval=1e-8):
    alphas = th.ones(N) / N
    if min_entropy is None or max_entropy is None:
        dists = th.distributions.Dirichlet(th.ones(N) / N).sample((n_samples,))
    else:
        assert min_entropy is not None and max_entropy is not None
        dists = th.distributions.Dirichlet(alphas).sample((n_samples,))
        h_dists = -(dists * th.log2(dists)).sum(-1)
        dists = dists[th.logical_and(min_entropy < h_dists, h_dists < max_entropy)]
        while len(dists) < n_samples:
            new_dists = th.distributions.Dirichlet(alphas).sample((5000,))
            h_dists = -(new_dists * th.log2(new_dists)).sum(-1)
            while h_dists.max() < min_entropy:
                alphas *= 2
                new_dists = th.distributions.Dirichlet(alphas).sample((5000,))
                h_dists = -(new_dists * th.log2(new_dists)).sum(-1)
            new_dists = new_dists[th.logical_and(min_entropy < h_dists, h_dists < max_entropy)]
            dists = th.cat([dists, new_dists], dim=0)

        dists = dists[:n_samples]

    if (dists.min(-1)[0] < minval).any():
        dists += minval
        dists /= dists.sum(-1)

    return dists


def sample_distance_matrix(m, n, d=32, max_dist=1.):
    """
    @param m: Number of source vectors
    @param n: Number of sink vectors
    @param d: Dimension of vectors
    @param max_dist: Diameter of the space
    @return: mxn distance matrix
    """
    max_norm = max_dist / 2
    unit_normal = th.distributions.MultivariateNormal(
        loc=th.zeros(d), covariance_matrix=th.eye(d))
    vecs_a = unit_normal.sample((m,))
    vecs_a = F.normalize(vecs_a, dim=-1)
    vecs_a *= max_norm

    vecs_b = unit_normal.sample((n,))
    vecs_b = F.normalize(vecs_b, dim=-1)
    vecs_b *= max_norm

    dist = (vecs_a.unsqueeze(1) - vecs_b).norm(p=2, dim=-1)

    return dist
