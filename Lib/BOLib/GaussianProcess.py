import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import time
import random

from Lib.BOLib.Kernel import Kernel


def wrapper_maximize_lilelihood(args):
    return maximize_lilelihood(*args)


def maximize_lilelihood(init_value, Kernel, x, y):
    start = time.time()

    def cost_and_grad(hparams, Kernel, x, y, return_grad):
        Kernel.hparams = hparams
        noise = 10**-8
        K = Kernel.get_newK(x)
        Cn = K + noise * np.identity(len(K))
        Cn_inv = np.linalg.inv(Cn)

        value = (
            0.5 * np.sum(np.log(eigh(Cn)[0]))
            + 0.5 * np.dot(np.dot(y, Cn_inv), y.T)
            + y.shape[0] / 2 * np.log(2 * np.pi)
        )

        gradK = Kernel.get_gradK(x, Cn)

        grad = np.zeros(len(Kernel.hparams))
        for i in range(len(Kernel.hparams)):
            grad[i] = 0.5 * np.trace(np.dot(Cn_inv, gradK[i])) - 0.5 * np.dot(
                np.dot(np.dot(np.dot(y, Cn_inv), gradK[i]), Cn_inv),
                y.reshape(-1, 1),
            )

        if return_grad == True:
            return value, grad
        else:
            return value

    result = minimize(
        x0=init_value,
        fun=lambda hparams: cost_and_grad(
            hparams=hparams,
            Kernel=Kernel,
            x=x,
            y=y,
            return_grad=True,
        ),
        jac=True,
        bounds=Kernel.bounds,
    )
    print("elapsed_time = ", time.time() - start)
    return result


class GaussianProcess:
    def __init__(self, option, hparams, bounds, noise=10**-8):
        self.noise = noise
        self.Cn = 0
        self.Cn_inv = 0
        self.Kernel = Kernel(option, hparams, bounds)

    def fit(self, x, y, optimize_hparams=False, nrand=24, ncores=24):
        if optimize_hparams:
            hparams_old = np.copy(self.Kernel.hparams)
            result = maximize_lilelihood(self.Kernel.hparams, self.Kernel, x, y)
            min_fun = result.fun
            if result.success:
                self.Kernel.hparams = result.x

            else:
                self.Kernel.hparams = hparams_old
            print(result)

            start = time.time()
            if nrand > 0:
                if ncores == 1:
                    print("single process random maximize likelihood")
                    for i in range(nrand):
                        start = time.time()

                        hparams_current = np.empty(self.Kernel.hparams.shape[0])
                        for j in range(self.Kernel.hparams.shape[0]):
                            hparams_current[j] = 10 ** -random.uniform(0, 10)
                        result = maximize_lilelihood(hparams_current, self.Kernel, x, y)
                        print(result)

                    if min_fun > result.fun:
                        self.Kernel.hparams = result.x
                        print("min_fun > result.fun")

                elif ncores > 1:
                    print("multi process random maximize likelihood")
                    print("nproc =", ncores)
                    print("nrand =", nrand)

                    values = []
                    for i in range(nrand):
                        hparams_current = np.empty(self.Kernel.hparams.shape[0])
                        for j in range(self.Kernel.hparams.shape[0]):
                            hparams_current[j] = 10 ** -random.uniform(0, 10)

                        values.append((hparams_current, self.Kernel, x, y))

                    from multiprocessing import Pool

                    p = Pool(ncores)
                    results = p.map(wrapper_maximize_lilelihood, values)
                    for result in results:
                        print(result)
                        if min_fun > result.fun:
                            print("min_fun > result.fun")
                            self.Kernel.hparams = result.x

                else:
                    raise ValueError("")
                print("elapsed_time =", time.time() - start)

        start = time.time()

        self.Cn = self.Kernel.get_newK(x) + self.noise * np.identity(x.shape[0])

        self.Cn_inv = np.linalg.inv(self.Cn)

        print("etime_GP.Cn =", str(time.time() - start)[:5])

    def predict(self, x, y, candidate, m0=None):
        mean = np.empty(candidate.shape[0])
        variance = np.empty(candidate.shape[0])

        if not m0 is None:
            y = y - m0[1]
        k = self.Kernel.get_k_v2(x, candidate)
        start = time.time()
        kCn_inv = np.dot(k, self.Cn_inv)

        mean = np.dot(kCn_inv, y)

        kxx = self.Kernel.get_kxx(candidate)

        variance = kxx - np.sum(kCn_inv * k, 1)
        if not m0 is None:
            mean = mean + m0[0]

        return mean, variance

    def make_nextCn(self, x, next_x):
        noise = self.noise
        Cn = self.Cn

        k = self.Kernel.get_k(x, next_x)
        next_Cn = np.empty([x.shape[0] + 1, x.shape[0] + 1])
        next_Cn[: x.shape[0], : x.shape[0]] = Cn
        next_Cn[-1, :-1] = k
        next_Cn[:-1, -1] = k
        next_Cn[-1, -1] = self.Kernel(next_x, next_x) + noise

        self.Cn = next_Cn


def cost_and_grad(hparams, Kernel, x, y, return_grad):
    Kernel.hparams = hparams
    noise = 10**-8
    K = Kernel.get_newK(x)
    Cn = K + noise * np.identity(len(K))

    Cn_inv = np.linalg.inv(Cn)

    value = (
        0.5 * np.sum(np.log(eigh(Cn)[0]))
        + 0.5 * np.dot(np.dot(y, Cn_inv), y.T)
        + y.shape[0] / 2 * np.log(2 * np.pi)
    )

    gradK = Kernel.get_gradK(x, Cn)

    grad = np.zeros(len(Kernel.hparams))
    for i in range(len(Kernel.hparams)):
        grad[i] = 0.5 * np.trace(np.dot(Cn_inv, gradK[i])) - 0.5 * np.dot(
            np.dot(np.dot(np.dot(y, Cn_inv), gradK[i]), Cn_inv), y.reshape(-1, 1)
        )

    if return_grad == True:
        # print(value)
        return value, grad
    else:
        return value
