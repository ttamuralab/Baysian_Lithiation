import numpy as np


atm_weight = {"H": 1.0, "Li": 6.968, "O": 16.0}


class poscar:
    def __init__(self, cfname):
        hunit, h, catm, natm, ra, ifDinamics = read_POSCAR(cfname)

        self.hunit = hunit
        self.h = h
        self.catm = catm
        self.natm = natm
        self.ra = ra
        self.ifDinamics = ifDinamics

        self.cfname = cfname

        self.h_inv = np.linalg.inv(h)
        self.natm_all = sum(natm)
        self.nkatm = len(catm)
        self.katm = get_katm(catm, natm)
        self.ta = get_ta(self.ra, self.hunit, self.h)

    def print_state(self):
        print()
        print("state of poscar class")
        print("  hunit =", self.hunit)
        for i in range(3):
            print(f"  h[{i}]  =", self.h[i])
        print("  catm  =", end=" ")
        for i in range(self.nkatm):
            print(self.catm[i], end=" ")
        print()
        print("  natm  =", end=" ")
        for i in range(self.nkatm):
            print(self.natm[i], end=" ")
        print()
        print()

    def show_by_ovito(self, cfname=None):
        import subprocess

        if cfname is None:
            cfname = self.cfname
        command = "open -a ovito " + cfname
        subprocess.run(command.split())

    def write(self, cfname, comment="Comment for POSCAR file"):
        write_POSCAR(cfname, self.hunit, self.h, self.catm, self.natm, self.ra, comment)

    def rotate(self, a, c, theta):
        torad = np.pi / 180

        a = a * torad  # 0~180
        c = c * torad  # 0~360
        theta = theta * torad  # 0~180

        n = np.array([0, 0, 1])

        rx = np.array(
            [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        )

        rz = np.array(
            [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]]
        )

        for i in range(self.natm_all):
            self.ra[i] = np.dot(rx, self.ra[i])
            self.ra[i] = np.dot(rz, self.ra[i])

        n = np.dot(rx, n)
        n = np.dot(rz, n)

        rn = np.empty([3, 3])
        rn[0][0] = n[0] ** 2 * (1 - np.cos(theta)) + np.cos(theta)
        rn[0][1] = n[0] * n[1] * (1 - np.cos(theta)) - n[2] * np.sin(theta)
        rn[0][2] = n[0] * n[2] * (1 - np.cos(theta)) + n[1] * np.sin(theta)

        rn[1][0] = n[0] * n[1] * (1 - np.cos(theta)) + n[2] * np.sin(theta)
        rn[1][1] = n[1] ** 2 * (1 - np.cos(theta)) + np.cos(theta)
        rn[1][2] = n[1] * n[2] * (1 - np.cos(theta)) - n[0] * np.sin(theta)

        rn[2][0] = n[0] * n[2] * (1 - np.cos(theta)) - n[1] * np.sin(theta)
        rn[2][1] = n[1] * n[2] * (1 - np.cos(theta)) + n[0] * np.sin(theta)
        rn[2][2] = n[2] ** 2 * (1 - np.cos(theta)) + np.cos(theta)

        # *nベクトル周りの回転
        for i in range(self.natm_all):
            self.ra[i] = np.dot(rn, self.ra[i])

        self.ta = get_ta(self.ra, self.hunit, self.h)

    def add_mol(self, mol, coodinate):
        ra_katm = []
        base_or_mol = []
        mol.adjust_g_to_0()
        if coodinate.shape[0] == 3:
            pass
        elif coodinate.shape[0] == 6:
            mol.rotate(coodinate[3], coodinate[4], coodinate[5])
        mol.ta = mol.ta + coodinate[:3]
        for ia in range(mol.natm_all):
            mol.ra[ia] = np.dot(self.h_inv, mol.ta[ia])

        for i in range(len(self.catm)):
            start = int(np.sum(self.natm[:i]))
            end = int(np.sum(self.natm[: i + 1]))
            ra_katm.append(self.ra[start:end])
            base_or_mol.append([0] * self.natm[i])

        for i in range(len(mol.catm)):
            if mol.catm[i] in self.catm:
                self.natm[self.catm.index(mol.catm[i])] += mol.natm[i]
            else:
                self.catm.append(mol.catm[i])
                self.natm.append(mol.natm[i])
                ra_katm.append(None)
                base_or_mol.append([])

        for i in range(len(self.catm)):
            for j in range(mol.ra.shape[0]):
                if self.catm[i] == mol.catm[mol.katm[j]]:
                    ra_appended = mol.ra[j]
                    # *周期境界
                    for id in range(3):
                        if ra_appended[id] < 0:
                            ra_appended[id] += 1
                        elif ra_appended[id] > 1:
                            ra_appended[id] -= 1

                    if ra_katm[i] is None:
                        ra_katm[i] = ra_appended
                    else:
                        ra_katm[i] = np.block(
                            [
                                [ra_katm[i]],
                                [ra_appended],
                            ]
                        )
                    base_or_mol[i].append(1)

        ra = None
        for i in range(len(ra_katm)):
            if ra is None:
                ra = ra_katm[i]
            else:
                # print(ra)
                ra = np.block([[ra], [ra_katm[i]]])
        self.ra = ra
        self.ta = get_ta(self.ra, self.hunit, self.h)

    def parallel_shift(self, coodinate, method="t"):
        if coodinate.shape[0] != 3:
            print("coodinate.shape is not 3 in parallel_shift of poscar")
            exit()

        self.ta = self.ta + coodinate
        for ia in range(self.natm_all):
            self.ra[ia] = np.dot(self.h_inv, self.ta[ia])

    def adjust_g_to_0(self):
        self.parallel_shift(-cal_g(self))


def get_katm(catm, natm):
    katm = []

    for i in range(len(catm)):
        katm = katm + [i] * natm[i]

    return katm


def get_ta(ra, hunit, h):
    ta = np.empty(ra.shape)
    for ia in range(ra.shape[0]):
        ta[ia] = np.dot(hunit * h, ra[ia])

    return ta


def cal_g(mol):
    sum_atm_weight = 0
    g = np.zeros(3)
    for i in range(np.sum(mol.natm)):
        g += atm_weight[mol.catm[mol.katm[i]]] * mol.ra[i]
        sum_atm_weight += atm_weight[mol.catm[mol.katm[i]]]
    g = g / sum_atm_weight
    # print("g = ", g)
    return g


def cal_I(ra, a, c, katm, atm_weight):
    n = np.array([0, 0, 1])
    torad = np.pi / 180

    a = a * torad  # 0~90
    c = c * torad  # 0~360

    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])

    n = np.dot(rx, n)
    n = np.dot(rz, n)

    I = 0
    for ia in range(ra.shape[0]):
        dist = np.linalg.norm(ra[ia] - np.dot(ra[ia], n) * n)

        I += atm_weight[katm[ia]] * dist**2

    return I


def minimize_I(mol):
    min_I = 100

    for a in range(0, 901, 1):
        for c in range(0, 3600, 1):
            I = cal_I(mol.ra, a / 10, c / 10, mol.katm, atm_weight)
            if I < min_I:
                min_I = I
                min_I_a = a
                min_I_c = c

    a = -min_I_a / 10
    c = -min_I_c / 10
    torad = np.pi / 180

    a = a * torad  # 0~90
    c = c * torad  # 0~360

    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])

    for ia in range(mol.ra.shape[0]):
        mol.ra[ia] = np.dot(rz, mol.ra[ia])
        mol.ra[ia] = np.dot(rx, mol.ra[ia])

    return mol.ra


def write_POSCAR(cfname, hunit, h, catm, natm, ra, comment="Comment for POSCAR file"):
    file = open(cfname, "w")
    file.write(comment)
    file.close()

    with open(cfname, "a") as f_hundle:
        f_hundle.write("\n")
        f_hundle.write(str(hunit) + "\n")
        # *h
        np.savetxt(f_hundle, h, fmt="%.18f")

        # *catm
        for i in range(len(catm)):
            f_hundle.write(str(catm[i]))
            if i != len(catm) - 1:
                f_hundle.write(" ")
        f_hundle.write("\n")

        # *natm
        for i in range(len(natm)):
            f_hundle.write(str(natm[i]))
            if i != len(natm) - 1:
                f_hundle.write(" ")
        f_hundle.write("\n")

        # *Direct
        f_hundle.write("Direct\n")

        # *ra
        np.savetxt(f_hundle, ra, fmt="%.18f")


def read_POSCAR(cfname):
    comment = None
    hunit = None
    h = np.empty([3, 3])
    catm = None
    natm = None
    ra = None
    ifDinamics = None

    with open(cfname, mode="r") as file:
        data = file.readlines()
        for i in range(len(data)):
            data[i] = data[i].replace("\n", "")
        # *コメント(string)
        comment = data[0]

        # *hunit(float)
        hunit = float(data[1])

        # *h(numpy)
        for i in range(2, 5):
            data[i] = data[i].split()
            for j in range(3):
                h[i - 2, j] = float(data[i][j])

        # *catm(list_string)
        catm = data[5].split()

        # *natm(list_int)
        data[6] = data[6].split()
        for i in range(len(data[6])):
            data[6][i] = int(data[6][i])
        natm = data[6]

        # *raとDinamics(numpy)
        if data[7] == "Selective dynamics":
            for j in range(9, len(data)):
                data[j] = data[j].split()
            ra, ifDinamics = stringListToNumpy(data[9:], "Selective dynamics")
        elif data[7] == "Direct" or data[7] == "direct":
            for j in range(8, 8 + np.sum(natm)):
                data[j] = data[j].split()
            ra = stringListToNumpy(data[8 : 8 + np.sum(natm)], "Direct")

        return hunit, h, catm, natm, ra, ifDinamics


def stringListToNumpy(list, mode):
    if mode == "Selective dynamics":
        # list = float(list)
        array = np.array(list, dtype=object)
        ra = array[:, :3]
        ra = ra.astype(float)
        ifDynamics = array[:, 3:]
        return ra, ifDynamics
    elif mode == "Direct":
        array = np.array(list, dtype=object)
        array = array[:, :3]
        array = array.astype(float)
        return array
