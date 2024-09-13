import numpy as np
from scipy.stats import norm


def cal_UCB(m, sigma, nsample):
    UCB = m + 5 * np.sqrt(np.log(nsample) / nsample) * sigma
    return UCB


def cal_maxEI(sample_y, mean, sigma):
    y_max = np.max(sample_y)
    z = (mean - y_max) / sigma
    if sigma == 0:
        return 0
    else:
        return sigma * (z * norm.cdf(z, 0, 1) + norm.pdf(z, 0, 1))


def cal_minEI(min_y, mean, variance):
    try:
        std = np.sqrt(variance)
        z = (min_y - mean) / std
        return std * (z * norm.cdf(z, 0, 1) + norm.pdf(z, 0, 1))
    except:
        return 0


def cal_minEI_(min_y, mean, variance):
    variance = np.where(variance < 10**-10, 10**-10, variance)
    std = np.sqrt(variance)
    z = (min_y - mean) / std
    return std * (z * norm.cdf(z, 0, 1) + norm.pdf(z, 0, 1))


def cal_minEI_map(y, mean, var):
    min_y = np.min(y)

    def cal_minEI_(mean, variance):
        std = np.sqrt(variance)
        z = (min_y - mean) / std
        return std * (z * norm.cdf(z, 0, 1) + norm.pdf(z, 0, 1))

    acq = map(cal_minEI_, mean, var)
    acq = np.array(list(acq))
    return acq
