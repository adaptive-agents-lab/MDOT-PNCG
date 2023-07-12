
import torch as th


def entropy(P):
    """Compute the entropy of a probability distribution P, with the convention that 0log0=0."""
    tiny = th.finfo(P.dtype).tiny
    return -(P * (P+tiny).log2()).sum()


def kl_div(p, q):
    tiny = th.finfo(p.dtype).tiny
    return (p * ((p+tiny)/q).log()).sum()


def err_func(P, r, c, dist='l1'):
    r_P = P.sum(-1)
    c_P = P.sum(-2)
    if dist == 'l1':
        err_r = (r_P - r).norm(p=1, dim=-1).max()
        err_c = (c_P - c).norm(p=1, dim=-1).max()
        err = err_r + err_c
    elif dist == 'hilbert':
        ratio_r = (r_P / r).log()
        ratio_c = (c_P / c).log()
        err_r = ratio_r.max() - ratio_r.min()
        err_c = ratio_c.max() - ratio_c.min()
        err = err_r + err_c
    elif dist == 'bregman':
        err_r = kl_div(r, r_P) - r.sum() + r_P.sum()
        err_c = kl_div(c, c_P) - c.sum() + c_P.sum()
        err = err_r + err_c
    elif dist == 'kl':
        if th.allclose(r_P.sum(), th.ones_like(r_P.sum())) and th.allclose(c_P.sum(), th.ones_like(c_P.sum())):
            err_r = kl_div(r, r_P)
            err_c = kl_div(c, c_P)
            err = err_r + err_c
        else:
            raise ValueError("KL divergence requires marginals to sum to 1. Please use Bregman divergence instead.")
    else:
        raise ValueError

    return err, err_r, err_c, r_P, c_P
