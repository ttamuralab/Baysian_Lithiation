import numpy as np
import sys

# import poscar


def main(
    struct,
    fname_output,
    ifConst=False,
    cgstep=0,
    kpoints=(1, 1, 1),
    cutoff=300,
    PAO=None,
    is_plus_U=False,
):
    #!open
    file = open(fname_output, "w")

    file.write("Systemname  " + "base_and_mol\n")
    file.write("SystemLabel " + "base_and_mol\n\n")
    file.write("NumberOfAtoms " + str(np.sum(struct.natm)) + "\n")
    file.write("NumberOfSpecies " + str(len(struct.natm)) + "\n")
    file.write("%block ChemicalSpeciesLabel\n")
    atomic_number = []
    for i in range(len(struct.natm)):
        if struct.catm[i] == "H":
            atomic_number.append(1)
        elif struct.catm[i] == "Li":
            atomic_number.append(3)
        elif struct.catm[i] == "C":
            atomic_number.append(7)
        elif struct.catm[i] == "O":
            atomic_number.append(8)
        elif struct.catm[i] == "Si":
            atomic_number.append(14)
        elif struct.catm[i] == "Ti":
            atomic_number.append(22)
        else:
            print("error in poscar_to_fdf.py")
            print(f"{struct.catm[i]} is undefined")
            exit()
    for i in range(len(struct.natm)):
        file.write(
            " " + str(i + 1) + " " + str(atomic_number[i]) + " " + struct.catm[i]
        )
        file.write("\n")
    file.write("%endblock ChemicalSpeciesLabel\n\n")

    if struct.catm[0] == "Ti" and struct.catm[1] == "O":
        file.write("%block PS.lmax\n")
        for i in range(len(struct.natm)):
            if struct.catm[i] == "Ti":
                file.write(" Ti 3\n")
            if struct.catm[i] == "O":
                file.write(" O 3\n")
        file.write("%endblock PS.lmax\n\n")

    if PAO == "DZP":
        file.write("PAO.BasisSize    DZP\n")
        file.write("PAO.EnergyShift     150 meV\n")
    else:

        file.write("PAO.BasisType    split\n")

        file.write("%block PAO.Basis\n")
        for i in range(len(struct.natm)):
            if struct.catm[i] == "Ti":
                file.write("Ti    5\n")

                file.write(" n=3    0    1   E     93.95      5.20\n")
                file.write("  5.69946662616249\n")
                file.write("  1.00000000000000\n")

                file.write(" n=3    1    1   E     95.47      5.20\n")
                file.write("  5.69946662616249\n")
                file.write("  1.00000000000000\n")

                file.write(" n=4    0    2   E     96.47      5.60\n")
                file.write("  6.09996398975307        5.09944363262274\n")
                file.write("  1.00000000000000        1.00000000000000\n")

                file.write(" n=3    2    2   E     46.05      4.95\n")
                file.write("  5.94327035784617        4.70009988294302\n")
                file.write("  1.00000000000000        1.00000000000000\n")

                file.write(" n=4    1    1   E      0.50      1.77\n")
                file.write("  3.05365979938936\n")
                file.write("  1.00000000000000\n")

            if struct.catm[i] == "O":
                file.write("O     3\n")

                file.write(" n=2    0    2   E     40.58      3.95\n")
                file.write("  4.95272270428712        3.60331408800389\n")
                file.write("  1.00000000000000        1.00000000000000\n")

                file.write(" n=2    1    2   E     36.78      4.35\n")
                file.write("  4.99990228025066        3.89745395068600\n")
                file.write("  1.00000000000000        1.00000000000000\n")

                file.write(" n=3    2    1   E     21.69      0.93\n")
                file.write("  2.73276990670788\n")
                file.write("  1.00000000000000\n")

        file.write("%endblock PAO.Basis\n\n")

    file.write("XC.Functional  GGA\n")
    file.write("XC.Authors     PBE\n\n")

    if is_plus_U:
        file.write("LDAU.ProjectorGenerationMethod 2	# default:2\n")
        file.write("%block LDAU.Proj      # Define LDAU projectors\n")
        file.write("Ti    1               # Label, l_shells\n")
        file.write(
            "  n=3 2 	                  # n (opt if not using semicore levels),l,Softconf(opt)\n"
        )
        file.write("  5.00  0.00          # U(eV), J(eV) for this shell\n")
        file.write(
            "  2.30  0.15          # rc(Bohr), \omega(Bohr) (Fermi cutoff function)\n"
        )
        file.write("%endblock LDAU.Proj\n\n")

    file.write("LatticeConstant       1.00 Ang\n")

    file.write("%block LatticeVectors\n")
    np.savetxt(file, struct.h, fmt="%.6f")
    file.write("%endblock LatticeVectors\n\n")

    file.write("%block kgrid_Monkhorst_Pack\n")
    file.write(f" {kpoints[0]} 0 0  0.\n")
    file.write(f" 0 {kpoints[1]} 0  0.\n")
    file.write(f" 0 0 {kpoints[2]}  0.\n")
    file.write("%endblock kgrid_Monkhorst_Pack\n\n")

    file.write(f"MeshCutoff          {cutoff} Ry \n\n")

    file.write("DM.MixingWeight      0.3\n")
    file.write("DM.NumberPulay       6\n")
    file.write("DM.Tolerance         1.d-4\n")
    file.write("DM.UseSaveDM      .true.\n\n")

    file.write("SolutionMethod       diagon\n")
    file.write("ElectronicTemperature  25 meV\n\n")

    file.write("MD.TypeOfRun         cg\n")
    file.write(f"MD.NumCGsteps        {cgstep}\n")
    file.write("MD.MaxCGDispl         0.1  Ang\n")
    file.write("MD.MaxForceTol        0.04 eV/Ang\n\n")

    file.write("AtomicCoordinatesFormat Fractional\n")
    file.write("%block AtomicCoordinatesAndAtomicSpecies\n")
    katm = []
    for i in range(len(struct.natm)):
        for j in range(struct.natm[i]):
            katm.append(i + 1)

    np.savetxt(
        file,
        np.block([struct.ra, np.array(katm).reshape(-1, 1)]),
        fmt=["%.18f", "%.18f", "%.18f", "%.i"],
    )
    file.write("%endblock AtomicCoordinatesAndAtomicSpecies")

    if ifConst:
        file.write("\n\n%block Geometry.Constraints\n")
        file.write("  atom [  1 --  24, 61 -- 108]\n")
        file.write("%endblock")

    file.close()


if __name__ == "__main__":
    icfname = sys.argv[1]
    ocfname = sys.argv[2]
