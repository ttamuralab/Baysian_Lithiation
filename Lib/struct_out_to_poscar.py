import numpy as np
import sys

from Lib import poscar

def main(icfname, ocfname):
    hunit = 1
    h = np.empty([3, 3])

    with open(icfname, mode="r") as file:
        data = file.readlines()

    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")

    # *h
    for i in range(3):
        data[i] = data[i].split()
        for j in range(3):
            h[i, j] = float(data[i][j])

    natms = int(data[3])
    print(natms)

    # *nkatm
    for i in range(4, 4 + natms):
        data[i] = data[i].split()

    nkatm = int(data[3 + natms][0])
    print(nkatm)

    # *natm
    natm = [0] * nkatm
    for i in range(natms):
        for j in range(nkatm):
            if int(data[4 + i][0]) == j + 1:
                natm[j] += 1
    print(natm)

    # *catm
    catm = []
    for i in range(nkatm):
        next = 0
        for j in range(i):
            next += natm[j]
        # print(next)
        element_number = int(data[4 + next][1])

        if element_number == 22:
            catm.append("Ti")
        elif element_number == 8:
            catm.append("O")
        elif element_number == 1:
            catm.append("H")
        elif element_number == 3:
            catm.append("Li")
        elif element_number == 14:
            catm.append("Si")
        else:
            print("error")
            sys.exit()
    print(catm)

    # pbc
    ra = np.array(data[4:])[:, 2:].astype(float)
    for i in range(natms):
        for j in range(3):
            if ra[i][j] < 0:
                ra[i][j] = ra[i][j] + 1
            elif ra[i][j] > 1:
                ra[i][j] = ra[i][j] - 1

    poscar.write_POSCAR(ocfname, hunit, h, catm, natm, ra)


if __name__ == "__main__":
    pass
