import numpy as np
import time
import datetime
import os

from Lib.BOLib.GaussianProcess import GaussianProcess
from Lib.BOLib import Acquisition


class BayesianOptimization:
    def __init__(self, option, beta, bounds, m0=None, candidate_label=None):
        self.x_label = candidate_label

        self.GP = GaussianProcess(option, beta, bounds)
        self.total_time = np.empty(0)
        self.each_time = np.empty(0)

    def one_step_run(
        self, feature, target, candidate, optimize_hparams=False, min_y=None, m0=None
    ):
        x = feature
        y = target
        if min_y is None:
            min_y = np.min(y)

        start = time.time()
        self.GP.fit(x, y, optimize_hparams)
        print("etime_GP.fit =", str(time.time() - start)[:5])

        start = time.time()
        mean, variance = self.GP.predict(x, y, candidate, m0=m0)
        print("etime_meanvar =", str(time.time() - start)[:5])

        start = time.time()

        acq = Acquisition.cal_minEI_(min_y, mean, variance)

        print("etime_acq =", str(time.time() - start)[:5])

        next_x = candidate[np.argmax(acq)]
        print("max_acq =", np.max(acq))
        print("argmax_acq =", np.argmax(acq))

        if np.max(acq) == 0:
            print("warning: max acq is zero")

        if np.isnan(np.max(acq)):
            print("np.max(acq) is None")
            for i in range(acq.shape[0]):
                if np.isnan(acq[i]):
                    print(i, mean[i], variance[i], acq[i], min_y)

            print(min_y)
            print(np.argmax(acq))
            print(np.max(acq))
            np.savetxt(
                "meanver.dat",
                np.block([mean.reshape(-1, 1), variance.reshape(-1, 1)]),
            )
            exit()

        status = {}
        status["max_acq"] = np.max(acq)
        status["next_feature"] = next_x
        status["mean"] = mean
        status["hparams"] = self.GP.Kernel.hparams
        status["var"] = variance

        return np.argmax(acq), status

    def multi_step_run(self, candidate, feature, n_search, mean, std, path, savefmt):
        self.y_mean = mean
        self.y_std = std
        self.y = (self.y - self.y_mean) / self.y_std

        start_time = time.time()

        i_search = 0
        self.candidate = candidate
        self.feature = (feature - self.y_mean) / self.y_std

        # * ループ管理
        while True:
            if n_search is None:  # *無限探索
                if np.min(self.y) == (np.min(feature) - self.y_mean) / self.y_std:
                    print("minimum y detected")
                    print("total search =", i_search)
                    break
            else:
                if i_search < n_search:  # *有限探索
                    if self.min_y == np.min(feature):
                        self.x = np.block([[self.x], [self.x[-1]]])

                        self.y = np.append(self.y, self.min_y)
                        i_search += 1
                        continue
                else:
                    break
            print()
            print(i_search + 1, "回目の探索")

            # * Main
            if i_search % 10 == 0:
                optimize_hparams = True
            else:
                optimize_hparams = False

            next_index, next_x = self.one_step_run(self.candidate, optimize_hparams)
            next_y = self.feature[next_index]

            self.GP.make_nextCn(self.x, next_x)

            self.x = np.block([[self.x], [next_x]])
            self.y = np.append(self.y, next_y)

            if next_y < self.min_y:
                self.min_y = next_y

            # *次回探索点を候補点から削除
            self.candidate = np.delete(self.candidate, next_index, 0)
            self.feature = np.delete(self.feature, next_index, 0)

            # * output
            print("  next search point:")
            # print("    x = ", next_x)
            print("    y = ", next_y * self.y_std + self.y_mean)
            print("  now minimum:")
            print("    min_y = ", self.min_y * self.y_std + self.y_mean)

            i_search += 1

            if i_search == 1:
                self.each_time = np.append(self.each_time, time.time() - start_time)
            else:
                self.each_time = np.append(
                    self.each_time, time.time() - start_time - self.total_time[-1]
                )
            self.total_time = np.append(self.total_time, time.time() - start_time)
            self.save_data(path, start_time, i_search, savefmt)
        print()

    def get_data(self):
        # 標準化の解除
        y = self.y * self.y_std + self.y_mean
        min_ys = np.empty(0)
        for i in range(self.x.shape[0]):
            min_ys = np.append(min_ys, np.min(y[: i + 1]))
        data = np.block([self.x, y.reshape(-1, 1), min_ys.reshape(-1, 1)])

        return data

    def save_data(self, path, start_time, i_search, fmt):
        data = self.get_data()
        os.makedirs(path, exist_ok=True)
        elapsed_time = "{:.2f}".format(time.time() - start_time)
        dt = datetime.datetime.fromtimestamp(start_time)
        fname = "miny_" + str(dt.strftime("%Y%m%d%H%M%S")) + ".dat"

        np.savetxt(
            path + "/" + fname,
            data,
            fmt=fmt,
            header=elapsed_time + "s" + "\n" + "seatch times =" + str(i_search),
        )

        path = path + "/time"
        os.makedirs(path, exist_ok=True)
        fname = "time_" + str(dt.strftime("%Y%m%d%H%M%S")) + ".dat"
        data = np.block([self.each_time.reshape(-1, 1), self.total_time.reshape(-1, 1)])

        np.savetxt(
            path + "/" + fname,
            data,
            fmt=["%f", "%f"],
        )

    def delete_near_candidate(self, next_x, norm):
        delete_index = np.empty(0)
        for i in range(self.candidate.shape[0]):
            x = np.abs(next_x - self.candidate[i])
            # 周期境界条件
            if x[0] > 4:
                x[0] = 8 - x[0]
            if x[1] > 4:
                x[1] = 8 - x[1]
            if x[4] > 2:
                x[4] = 4 - x[4]

            if np.linalg.norm(x) <= norm:
                delete_index = np.append(delete_index, int(i))

        self.candidate = np.delete(self.candidate, delete_index.astype(int).tolist(), 0)
        self.feature = np.delete(self.feature, delete_index.astype(int).tolist(), 0)
