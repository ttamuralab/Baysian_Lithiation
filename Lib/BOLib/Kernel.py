import numpy as np
import scipy.spatial.distance as distance


class Kernel:
    def __init__(self, option, hparams, bounds):
        self.option = option
        self.hparams = hparams
        self.bounds = bounds

    def __call__(self, x1, x2):
        kernel = 0
        print("called")
        for i in range(len(self.hparams)):
            if self.option[i][0] == "rbf":
                kernel = kernel + rbf(x1[i], x2[i], self.hparams[i])
            elif self.option[i][0] == "pbc":
                kernel = kernel + pbc(x1[i], x2[i], self.option[i][1], self.hparams[i])
            else:
                print("error in Kernel__call__")
                exit()

        return np.exp(kernel)

    def get_gradK(self, x, Cn):
        gradK = np.zeros([len(self.hparams), x.shape[0], x.shape[0]])
        for i in range(len(self.hparams)):
            xi = x[:, i]
            gradK[i] = -((xi - xi[:, np.newaxis]) ** 2)
            gradK[i] = gradK[i] * Cn
        return gradK

    def get_newK(self, x):
        if x.ndim == 1:
            K = np.zeros([x.shape[0], x.shape[0]])
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    K[i][j] = np.exp(rbf(x[i], x[j], self.hparams[0]))
            return K

        hparams = self.hparams
        for i in range(len(hparams)):
            if hparams[i] < 10**-10:
                hparams[i] = 10**-10
        v = np.reciprocal(hparams)
        dist = distance.pdist(x, "seuclidean", V=v)
        dist = -np.square(dist)
        dist = np.exp(dist)
        K = distance.squareform(dist)
        K = K + np.identity(x.shape[0])
        return K

    def get_k(self, x, candidate):
        hparams = self.hparams
        for i in range(len(hparams)):
            if hparams[i] < 10**-10:
                hparams[i] = 10**-10
        k = np.zeros(x.shape[0])
        for i in range(k.shape[0]):
            if x.ndim == 1:
                k[i] = np.exp(rbf(x[i], candidate, self.hparams[0]))
            else:
                v = np.reciprocal(hparams)
                dist = distance.seuclidean(x[i], candidate, v)
                dist = -np.square(dist)
                k[i] = np.exp(dist)
        # print(k.shape)
        return k

    def get_k_v2(self, x, candidate):
        hparams = self.hparams

        k = candidate[:, np.newaxis, :] - x[np.newaxis, :, :]
        k = k * np.sqrt(hparams)
        k = np.linalg.norm(k, axis=2, ord=2)
        k = np.exp(-np.square(k))
        # print(k.shape)
        return k

    def get_kxx(self, x):
        hparams = self.hparams
        kxx = x - x
        kxx = kxx * np.sqrt(hparams)
        kxx = np.linalg.norm(kxx, axis=1, ord=2)
        kxx = np.exp(-np.square(kxx))
        # print(k.shape)
        return kxx


def rbf(x1, x2, beta):
    # sigma = 0.1
    # norm = np.linalg.norm(x1 - x2)
    return -beta * ((x1 - x2) ** 2)


def pbc(x1, x2, t, beta):
    # l = 0.05
    return -beta * (np.sin(np.pi * (x1 - x2) / t)) ** 2


def gaussianKernel(x1, x2):
    kernel = 1

    option = [
        ["pbc", 8],
        ["pbc", 8],
        ["rbf"],
        ["rbf"],
        ["pbc", 4],
        ["rbf"],
    ]
    beta = [2.00591813, 2.09461779, 0.01361527, 0.01361714, 0.0242104, 0.01361624]
    # beta = [0.75527672, 0.79017501, 0.00510014, 0.00703146, 0.06257186, 0.02623189]

    kernel = 0
    for i in range(6):
        if option[i][0] == "rbf":
            kernel = kernel + rbf(x1[i], x2[i], beta[i])
        elif option[i][0] == "pbc":
            kernel = kernel + pbc(x1[i], x2[i], option[i][1], beta[i])
        else:
            print("error")
            exit()

    return np.exp(kernel)


def rbf(x1, x2, beta):
    # sigma = 0.1
    norm = np.linalg.norm(x1 - x2)
    return -beta * (norm**2)
    # return np.exp(-(norm**2) / 2 / l**2)


def pbc(x1, x2, t, beta):
    # l = 0.05
    return -beta * (np.sin(np.pi * (x1 - x2) / t)) ** 2
    # return np.exp(-1 / 2 * (np.sin(np.pi * (x1 - x2) / t) / l) ** 2)
