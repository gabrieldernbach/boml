import torch
# VMap does not seem to be importable
#from torch import vmap
from gpytorch.kernels import Kernel


class AnovaKernel(Kernel):
    has_lengthscale = False
    is_stationary = False

    def __init__(self, m, s, *args, **kwargs):
        self.m = m
        self.s = s
        super(AnovaKernel, self).__init__(*args, **kwargs)

    def forward(self, x1, x2, **params):
        def single_eval(x1, x2):
            is_computed = torch.zeros((self.m+1, self.s + 1))
            res = torch.empty((self.m + 1, self.s + 1), device=x1.device)

            is_computed[:, 0] = 1
            res[:, 0] = 1

            def rec_func(m, s):
                if not is_computed[m, s]:
                    print(f'm:{m} s:{s}')
                    if s == 0 and m >= 0:
                        res[m, s] = 1
                        is_computed[m, s] = 1
                        return 1
                    if m < s:
                        res[m, s] = 0
                        is_computed[m, s] = 1
                        return 0
                    cres = x1[m] * x2[m] * rec_func(m-1, s-1) + rec_func(m-1, s)
                    res[m, s] = cres
                    is_computed[m, s] = 1
                    return cres

                else:
                    return res[m, s]
            result = rec_func(self.m, self.s)
            return result
        kernel_eval = vmap(vmap(single_eval))
        return kernel_eval(x1, x2)


class MultilinearKernel(Kernel):
    pass
