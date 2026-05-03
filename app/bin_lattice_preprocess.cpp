#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>

#include "NTL/LLL.h"

#include "../include/lattice.h"

int main(int argc, char **argv)
{
    int help = 0;
    int qp_lll = 1;
    int verbose = 0;
    const char *input = NULL;
    const char *output = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcasecmp(argv[i], "-h") || !strcasecmp(argv[i], "--help")) {
            help = 1;
        } else if (!strcasecmp(argv[i], "--no-qp-lll")) {
            qp_lll = 0;
        } else if (!strcasecmp(argv[i], "-v") || !strcasecmp(argv[i], "--verbose")) {
            verbose = 1;
        } else if (!input) {
            input = argv[i];
        } else if (!output) {
            output = argv[i];
        } else {
            help = 1;
        }
    }

    if (help || !input || !output) {
        fprintf(stderr, "Usage: %s <input_basis> <output_basis> [--no-qp-lll] [-v]\n", argv[0]);
        fprintf(stderr, "Reads an NTL-format integer basis, runs exact NTL LLL, and writes a reduced basis for bgj_epi8.\n");
        return help ? 0 : 2;
    }

    NTL::Mat<NTL::ZZ> L_ZZ;
    std::ifstream data(input, std::ios::in);
    if (!data) {
        fprintf(stderr, "Error: cannot open input basis: %s\n", input);
        return 1;
    }
    data >> L_ZZ;
    if (L_ZZ.NumRows() == 0 || L_ZZ.NumCols() == 0) {
        fprintf(stderr, "Error: incorrect input basis format: %s\n", input);
        return 1;
    }

    NTL::ZZ det2;
    LLL(det2, L_ZZ, 1, 3, verbose ? 1 : 0);

    Lattice_QP L(L_ZZ);
    if (qp_lll) {
        L.compute_gso_QP();
        L.size_reduce();
        L.LLL_QP(0.99);
        L.to_int();
    }
    L.store(output);
    return 0;
}
