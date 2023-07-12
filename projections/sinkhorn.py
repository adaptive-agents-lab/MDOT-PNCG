
import torch as th

from projections import BaseProjector
from utils.measurements import err_func


class SinkhornProjector(BaseProjector):
    def project(self, P_0, r, c, eps, u, v):
        """Project the matrix P_0 onto U(r, c) with error epsilon using Sinkhorn iteration."""
        B, N, _ = P_0.size()
        assert N == _  # WLOG
        logs = {
            "errs": [],
            "errs_r": [],
            "errs_c": [],
            "duals": [],
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

        while (err > eps and k < self.maxIter) or k < self.minIter:
            k += 1
            delta_u = th.zeros_like(u)
            delta_v = th.zeros_like(v)
            if (k % 2) == 1:
                delta_u = r.log() - r_P.log()
                u += delta_u
            else:
                delta_v = c.log() - c_P.log()
                v += delta_v

            P *= delta_u.exp().unsqueeze(-1) * delta_v.exp().unsqueeze(-2)
            if P.isnan().any():
                raise ValueError("NaNs encountered in P")

            err, err_r, err_c, r_P, c_P = err_func(P, r, c, dist=self.sm)
            logs["errs"].append(err)
            logs["errs_r"].append(err_r)
            logs["errs_c"].append(err_c)

        if u.isnan().any() or v.isnan().any():
            raise ValueError

        logs["n_iter"] = k
        logs["kappa"] = kappa

        return P, u, v, logs
