import timeit
import torch
# VMap does not seem to be importable
# from torch import vmap
from gpytorch.kernels import Kernel


def anova_kernel(P, X, M, d):
    cache = dict()

    def fetch(P, X, j, t):
        if (j, t) not in cache:
            if t == 0:
                cache[(j, t)] = 1.
            elif j < t:
                cache[(j, t)] = 0.
            else:
                cache[(j, t)] = (
                    fetch(P, X, j-1, t)
                    + P[j, :, None]
                    * X.T[j, None, :]
                    * fetch(P, X, j-1, t-1))
        return cache[(j, t)]
    return fetch(P, X, d-1, M-1).sum(0, keepdims=True).T


def anova_batch_batchwise(x1, x2, degree):
    dim = x1.shape[1]
    res = torch.empty((x1.shape[0], x2.shape[0], dim, degree + 1), device=x1.device)
    res[..., 0] = 1
    xz = x1[:, None, :] * x2[None, ...]
    for cdeg in range(degree):
        deg = cdeg + 1
        rolled = torch.roll(res[..., cdeg], 1, dims=-1)
        rolled[..., :cdeg] = 0
        cur = xz * rolled
        cres = torch.cumsum(cur, dim=2)
        res[..., deg] = cres

    return res[:, :, -1, -1]


def anova_single_batchwise(x1, x2, degree):
    assert x1.shape[0] == x2.shape[0]
    dim = x1.shape[0]
    res = torch.empty((dim, degree + 1), device=x1.device)
    res[:, 0] = 1
    xz = x1 * x2
    for cdeg in range(degree):
        deg = cdeg + 1
        rolled = torch.roll(res[:, cdeg], 1)
        rolled[:cdeg] = 0
        cur = xz * rolled
        cres = torch.cumsum(cur, dim=0)
        res[:, deg] = cres

    return res[-1, -1]


def anova_single_eval_matrix(x1, x2, degree):
    assert x1.shape[0] == x2.shape[0]
    dim = x1.shape[0]
    is_computed = torch.zeros((dim, degree))
    res = torch.empty((dim, degree), device=x1.device)

    def rec_func(m, s):
        # Actually we're treating m and s as m+1 and s+1
        if not is_computed[m, s]:
            if s < 0:
                return 1
            if m < s:
                if s >= 0 and m >= 0:
                    res[m, s] = 0
                    is_computed[m, s] = 1
                return 0
            cres = x1[m] * x2[m] * rec_func(m-1, s-1) + rec_func(m-1, s)
            res[m, s] = cres
            is_computed[m, s] = 1
            return cres

        else:
            return res[m, s]
    result = rec_func(dim - 1, degree - 1)
    return result


class AnovaKernel(Kernel):
    has_lengthscale = False
    is_stationary = False

    def __init__(self, s, m, *args, **kwargs):
        self.degree = m  # Degree of Multilinear Kernel
        super(AnovaKernel, self).__init__(*args, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        # TODO there has to be something quicker than for loops...
        if diag:
            pass
            # TODO
        K = anova_batch_batchwise(x1, x2, degree=self.degree)
        if diag:
            return torch.diag(K)
        return K


class MultilinearKernel(Kernel):
    is_stationary = False

    def __init__(self, dim, *args, **kwargs):
        super(MultilinearKernel, self).__init__(*args, **kwargs)
        self.dim = dim  # Degree of Multilinear Kernel
        self.gammas = torch.nn.parameter.Parameter(torch.ones((dim)), requires_grad=False)

    def forward(self, x1, x2, diag=False, **params):
        g_sq = self.gammas ** 2

        if diag:
            # TODO
            pass

        k_help = x1[:, None, :] * x2[None, ...]
        k_help += g_sq[None, None, :]
        k_help /= (1 + g_sq)[None, None, :]
        K = torch.prod(k_help, dim=-1)
        if diag:
            return torch.diag(K)
        return K
