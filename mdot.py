
import torch as th

from projections.sinkhorn import SinkhornProjector
from projections.conjugate_gradient import ConjugateGradientProjector
from utils.algorithmic import round_altschuler


def construct_projector(projector_name, **kwargs):
    if projector_name == 'Sinkhorn':
        return SinkhornProjector(**kwargs)
    elif projector_name == 'ConjugateGradient':
        return ConjugateGradientProjector(**kwargs)
    else:
        raise ValueError("Unknown projector: {}".format(projector_name))


def mdot(P0, C, r, c, eps, gamma, T, projection_kwargs, projection='Sinkhorn', warmstart=True):
    B, N, _ = P0.size()
    assert N == _  #WLOG

    default_projection_kwargs = {
        "maxIter": 10000,
        "stopping_measure": "kl",
    }
    default_projection_kwargs.update(projection_kwargs)
    projection_kwargs = default_projection_kwargs
    projector = construct_projector(projection, **projection_kwargs)

    u0 = th.zeros_like(r)
    v0 = th.zeros_like(c)
    u = u0.clone()
    v = v0.clone()

    MD_update = th.exp(-gamma * C).unsqueeze(0)
    P = P0.clone()

    logs = {
        "costs": [(P * C).sum().item()],
        "proj_logs": [],
        "init_errors": [],
        "ls_func_cnt": 0
    }

    t = 0
    while t < T:
        t += 1
        # Take gradient step in dual space and map back to primal space
        P_hat = P * MD_update
        # Approximately project onto U(r, c) using eps as the threshold
        if warmstart:
            P, u, v, proj_logs = projector.project(P_hat, r, c, eps, u, v)
        else:
            P, u, v, proj_logs = projector.project(P_hat, r, c, eps, u0.clone(), v0.clone())

        logs["ls_func_cnt"] += proj_logs.get("ls_func_cnt", 0)
        logs["proj_logs"].append(proj_logs)
        logs["costs"].append((P * C).sum().item())
        logs["init_errors"].append(proj_logs['errs'][0].item())

    k_total = sum([log["n_iter"] for log in logs["proj_logs"]])

    logs["rounded_cost"] = (round_altschuler(P, r, c).squeeze(0) * C).sum().item()

    return P, k_total, logs
