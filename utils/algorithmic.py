
import time
import torch as th

def eval_dual(P, u, v, r, c):
    return P.sum() - 1 - (u * r).sum() - (v * c).sum()
    # return P.sum().log() - (u * r).sum() - (v * c).sum()


def round_altschuler(P, r, c):
    X = th.min(r / P.sum(-1), th.ones_like(r))
    X = th.stack([th.diag(X[_]) for _ in range(len(X))])
    P = th.bmm(X, P)

    Y = th.min(c / P.sum(-2), th.ones_like(c))
    Y = th.stack([th.diag(Y[_]) for _ in range(len(Y))])
    P = th.bmm(P, Y)

    err_r = r - P.sum(-1)
    err_c = c - P.sum(-2)
    P += th.bmm(err_r.unsqueeze(-1), err_c.unsqueeze(-2)) / err_r.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1)

    return P


def dual_hessian(P, r_P, c_P):
    B, N, _ = P.size()
    H = th.zeros((B, 2 * N, 2 * N), device=P.device, dtype=P.dtype)
    # Fill the diagonal of H with the concatenation of the row and column sums of P
    H[:, :N, N:] = P
    H[:, N:, :N] = P.transpose(2, 1)
    H = th.diagonal_scatter(H, th.cat([r_P, c_P], dim=-1), dim1=-2, dim2=-1)

    return H


def secant_approx_wolfe(df, x0=0., x1=1., c1=1e-3, c2=0.5, max_iter=100):
    logs = {"f": {}, "df": {}}
    func_cnt = 0
    x1 = x1.item() if isinstance(x1, th.Tensor) else x1

    df0 = df(x0)
    df_x0 = df0.clone()
    logs["df"][x0] = df0
    func_cnt += 1

    df_x1 = df(x1)
    logs["df"][x1] = df_x1
    func_cnt += 1

    if (2 * c1 - 1) * df0 >= df_x1 >= c2 * df0:
        logs["func_cnt"] = func_cnt
        return x1, logs

    while df_x1 < 0:
        x0 = x1
        x1 *= 2
        df_x1 = df(x1)
        func_cnt += 1
        logs["df"][x1] = df_x1

    if df_x1.isnan():
        raise ValueError("NaN encountered.")

    while True:
        x = (x0 * df_x1 - x1 * df_x0) / (df_x1 - df_x0)
        x = (x + (x0 + x1) / 2) / 2
        df_x = df(x)
        func_cnt += 1
        logs["df"][x.item()] = df_x
        if df_x > 0:
            x1 = x
            df_x1 = df_x
        elif df_x < 0:
            x0 = x
            df_x0 = df_x

        if (2 * c1 - 1) * df0 >= df_x >= c2 * df0:
            break

        if func_cnt > max_iter:
            raise RuntimeError("Secant approximate Wolfe line search did not converge.")

    logs["func_cnt"] = func_cnt
    # print("Bisection took {:.6f} seconds and {}.".format(time.time() - start_time, k))

    return x, logs
