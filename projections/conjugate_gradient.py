
import torch as th

from projections import BaseProjector
from utils.algorithmic import dual_hessian, secant_approx_wolfe
from utils.measurements import err_func


class ConjugateGradientProjector(BaseProjector):
    def __init__(self, maxIter=1000, minIter=0, sm="l1", method="FR", descent_dir='Sinkhorn', **kwargs):
        super().__init__(maxIter, minIter, sm=sm, **kwargs)
        self.method = method
        self.descent_dir = descent_dir
        self.compute_psc = self._kwargs.get("compute_psc", False)

    def project(self, P_0, r, c, eps, u, v):
        B, N, _ = P_0.size()
        assert N == _  # WLOG
        logs = {
            "errs": [],
            "errs_r": [],
            "errs_c": [],
            "alphas": [],
            "alpha_avgs": [],
            "ls_logs": [],
            "betas": [],
            "descent_dir_norms": [],
            "psc_H": [],
            "psc_MH": [],
        }

        k = 0
        tmp = P_0 * u.exp().unsqueeze(-1) * v.exp().unsqueeze(-2)
        kappa = tmp.view(B, -1).norm(p=1, dim=-1).log().unsqueeze(-1)
        u -= kappa
        P = P_0 * u.exp().unsqueeze(-1) * v.exp().unsqueeze(-2)
        err, err_r, err_c, r_P, c_P = err_func(P, r, c, dist=self.sm)

        logs["errs"].append(err)
        logs["errs_r"].append(err_r)
        logs["errs_c"].append(err_c)
        grad_u_old, grad_v_old = None, None
        p_u, p_v = th.zeros_like(u), th.zeros_like(v)
        while (err > eps and k < self.maxIter) or k < self.minIter:
            k += 1

            grad_u = r_P - r
            grad_v = c_P - c
            if self.descent_dir == 'Sinkhorn':
                d_u = r_P.log() - r.log()
                d_v = c_P.log() - c.log()
            elif self.descent_dir == 'Gradient':
                d_u = grad_u
                d_v = grad_v
            else:
                raise ValueError("Unknown descent direction: " + self.descent_dir)

            if grad_v.isnan().any() or grad_u.isnan().any():
                raise ValueError

            if k == 1: # or (self.descent_dir == 'Gradient' and k % N == 0):
                beta = 0.
            else:
                grad_diff = th.cat([grad_u - grad_u_old, grad_v - grad_v_old], dim=-1)
                if self.descent_dir == 'Sinkhorn':
                    sink = th.cat([d_u, d_v], dim=-1)
                else:
                    sink = th.cat([grad_u, grad_v], dim=-1)
                p_old = th.cat([p_u, p_v], dim=-1)
                beta = self.compute_beta_PPR(grad_diff, sink, p_old)

            logs["betas"].append(beta)
            p_u = -d_u + beta * p_u
            p_v = -d_v + beta * p_v

            # Ensure descent direction, otherwise reset CG
            c_ = (p_u * r).sum() + (p_v * c).sum()
            if c_ - (p_u * r_P).sum() - (p_v * c_P).sum() < 0:
                p_u = -d_u
                p_v = -d_v
                c_ = (p_u * r).sum() + (p_v * c).sum()

            # BEGIN secant-bisection hybrid
            delta = p_u.unsqueeze(-1) + p_v.unsqueeze(-2)
            P_delta = P * delta
            df_fn = lambda a: (P_delta * (delta * a).exp()).sum() - c_
            alpha, ls_logs = secant_approx_wolfe(df_fn)
            # END secant-bisection hybrid

            logs["ls_logs"].append(ls_logs)
            logs["alphas"].append(alpha)

            delta_u = p_u * alpha
            delta_v = p_v * alpha

            # Projecting out component parallel to this eigenvector is not necessary
            # d_parallel = (delta_u.sum() - delta_v.sum()) / (2 * N)
            # delta_u -= d_parallel * th.ones_like(delta_u)
            # delta_v += d_parallel * th.ones_like(delta_v)

            u += delta_u
            v += delta_v

            grad_u_old = grad_u.clone()
            grad_v_old = grad_v.clone()

            P = P * delta_u.exp().unsqueeze(-1) * delta_v.exp().unsqueeze(-2)

            if P.isnan().any():
                raise ValueError

            err, err_r, err_c, r_P, c_P = err_func(P, r, c, dist=self.sm)

            if self.compute_psc:
                H = dual_hessian(P, r_P, c_P)
                if self.descent_dir == 'Sinkhorn':
                    M1 = (r.log() - r_P.log()) / (r - r_P)
                    M2 = (c.log() - c_P.log()) / (c - c_P)
                else:
                    raise ValueError
                M = th.diagonal_scatter(H, th.cat([M1, M2], dim=-1), dim1=-2, dim2=-1)
                MH = M @ H
                eig_H = th.linalg.eigvals(H)
                eig_MH = th.linalg.eigvals(MH)
                eig_H = th.sort(eig_H.real, descending=True)[0][0]
                eig_MH = th.sort(eig_MH.real, descending=True)[0][0]
                psc_H = eig_H[0] / eig_H[-2]
                psc_MH = eig_MH[0] / eig_MH[-2]
                logs["psc_H"].append(psc_H)
                logs["psc_MH"].append(psc_MH)

            logs["errs"].append(err)
            logs["errs_r"].append(err_r)
            logs["errs_c"].append(err_c)

        if u.isnan().any() or v.isnan().any():
            raise ValueError

        logs["n_iter"] = k
        logs["ls_func_cnt"] = sum([l["func_cnt"] for l in logs["ls_logs"]])

        return P, u, v, logs

    def compute_beta_PPR(self, gamma, sink, p_old):
        nom = (gamma * sink).sum()
        denom = (gamma * p_old).sum()

        beta = nom / denom

        return beta
