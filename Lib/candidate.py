import numpy as np


class Candidate:
    def __init__(self) -> None:
        self.cooang_r_unsearched = None
        self.cooang_r_searched = None

        self.cooang_t_unsearched = None
        self.cooang_t_searched = None

        self.descriptor_unsearched = None
        self.descriptor_searched = None

        self.feature_unsearched = None
        self.feature_searched = None

        self.n_all = 0
        self.n_searched = 0
        self.n_unsearched = 0

        self.m0_c = None
        self.m0_y = None
        self.y = None

        self.y_current = None
        self.cooang_t_searched_current = None

    def read(self, path):
        try:
            # *searched
            data = np.loadtxt(f"{path}/candidate_cooang_r_searched.dat")
            self.cooang_r_searched = data[:, :-1]
            self.y = data[:, -1]
            self.cooang_t_searched = np.loadtxt(
                f"{path}/candidate_cooang_t_searched.dat"
            )[:, :-1]
            self.feature_searched = np.loadtxt(f"{path}/candidate_feature_searched.dat")
            self.descriptor_searched = np.loadtxt(
                f"{path}/candidate_descriptor_searched.dat"
            )

            # *unsearched
            self.cooang_r_unsearched = np.loadtxt(
                f"{path}/candidate_cooang_r_unsearched.dat"
            )
            self.cooang_t_unsearched = np.loadtxt(
                f"{path}/candidate_cooang_t_unsearched.dat"
            )
            self.feature_unsearched = np.loadtxt(
                f"{path}/candidate_feature_unsearched.dat"
            )
            self.descriptor_unsearched = np.loadtxt(
                f"{path}/candidate_descriptor_unsearched.dat"
            )

            # *current
            data = np.loadtxt(f"{path}/candidate_cooang_t_searched_current.dat")
            self.cooang_t_searched_current = data[:, :-1]
            self.y_current = data[:, -1]

        except:
            self.cooang_r_unsearched = np.loadtxt(f"{path}/candidate_cooang_r_all.dat")
            self.cooang_t_unsearched = np.loadtxt(f"{path}/candidate_cooang_t_all.dat")
            self.feature_unsearched = np.loadtxt(f"{path}/candidate_feature_all.dat")
            self.descriptor_unsearched = np.loadtxt(
                f"{path}/candidate_descriptor_all.dat"
            )

        try:
            self.m0_y = np.loadtxt(f"{path}/m0_y.dat")
            self.m0_c = np.loadtxt(f"{path}/m0_c.dat")
        except:
            self.m0_y = np.empty(0)
            self.m0_c = np.zeros(self.cooang_r_unsearched.shape[0])

    def write(self, path):
        import os

        os.makedirs(path, exist_ok=True)

        # *unsearched
        np.savetxt(
            f"{path}/candidate_cooang_r_unsearched.dat",
            self.cooang_r_unsearched,
            fmt="%i",
        )
        np.savetxt(
            f"{path}/candidate_cooang_t_unsearched.dat",
            self.cooang_t_unsearched,
            fmt=["%f", "%f", "%f"],
        )
        np.savetxt(
            f"{path}/candidate_feature_unsearched.dat",
            self.feature_unsearched,
            fmt="%f",
        )
        np.savetxt(
            f"{path}/candidate_descriptor_unsearched.dat",
            self.descriptor_unsearched,
            fmt="%f",
        )

        # *searched
        if self.cooang_r_searched.ndim == 1:
            fmt_cooang_r = "%i"
            fmt_cooang_t = "%f"
            fmt_fearure = "%.18e"
            fmt_descriptor = "%.18e"

            np.savetxt(
                f"{path}/candidate_cooang_r_searched.dat",
                np.append(self.cooang_r_searched, self.y),
            )
            np.savetxt(
                f"{path}/candidate_cooang_t_searched.dat",
                np.append(self.cooang_t_searched, self.y),
            )
            np.savetxt(f"{path}/candidate_feature_searched.dat", self.feature_searched)
            np.savetxt(
                f"{path}/candidate_descriptor_searched.dat", self.descriptor_searched
            )
            np.savetxt(
                f"{path}/candidate_cooang_t_searched_current.dat",
                np.append(self.cooang_t_searched_current, self.y_current),
            )
        else:
            fmt_cooang_r = ["%i"] * self.cooang_r_searched.shape[1] + ["%f"]
            fmt_cooang_t = "%f"
            fmt_fearure = "%.18e"
            fmt_descriptor = "%.18e"

            np.savetxt(
                f"{path}/candidate_cooang_r_searched.dat",
                np.block([self.cooang_r_searched, self.y.reshape(-1, 1)]),
                fmt=fmt_cooang_r,
            )
            np.savetxt(
                f"{path}/candidate_cooang_t_searched.dat",
                np.block([self.cooang_t_searched, self.y.reshape(-1, 1)]),
                fmt=fmt_cooang_t,
            )
            np.savetxt(
                f"{path}/candidate_feature_searched.dat",
                self.feature_searched,
                fmt=fmt_fearure,
            )
            np.savetxt(
                f"{path}/candidate_descriptor_searched.dat",
                self.descriptor_searched,
                fmt=fmt_descriptor,
            )
            np.savetxt(
                f"{path}/candidate_cooang_t_searched_current.dat",
                np.block(
                    [self.cooang_t_searched_current, self.y_current.reshape(-1, 1)]
                ),
                fmt=fmt_cooang_t,
            )

        if not self.m0_c is None:
            np.savetxt(f"{path}/m0_y.dat", self.m0_y)
            np.savetxt(f"{path}/m0_c.dat", self.m0_c)

    def update(self, next_index, new_y):
        next_cooang_r = self.cooang_r_unsearched[next_index]
        next_cooang_t = self.cooang_t_unsearched[next_index]
        next_feature = self.feature_unsearched[next_index]
        next_descriptor = self.descriptor_unsearched[next_index]

        try:
            self.cooang_r_searched = np.block(
                [[self.cooang_r_searched], [next_cooang_r]]
            )
            self.cooang_t_searched = np.block(
                [[self.cooang_t_searched], [next_cooang_t]]
            )
            self.cooang_t_searched_current = np.block(
                [[self.cooang_t_searched_current], [next_cooang_t]]
            )
            self.feature_searched = np.block([[self.feature_searched], [next_feature]])
            self.descriptor_searched = np.block(
                [[self.descriptor_searched], [next_descriptor]]
            )

            self.y = np.block([self.y, new_y])
            self.y_current = np.block([self.y_current, new_y])
        except:
            self.cooang_r_searched = next_cooang_r
            self.cooang_t_searched = next_cooang_t
            self.cooang_t_searched_current = next_cooang_t
            self.feature_searched = next_feature
            self.descriptor_searched = next_descriptor

            self.y = new_y
            self.y_current = new_y

        self.cooang_r_unsearched = np.delete(self.cooang_r_unsearched, next_index, 0)
        self.cooang_t_unsearched = np.delete(self.cooang_t_unsearched, next_index, 0)
        self.feature_unsearched = np.delete(self.feature_unsearched, next_index, 0)
        self.descriptor_unsearched = np.delete(
            self.descriptor_unsearched, next_index, 0
        )

        try:
            self.m0_y = np.append(self.m0_y, self.m0_c[next_index])
            self.m0_c = np.delete(self.m0_c, next_index, 0)
        except:
            pass

    def delete(self):
        pass
