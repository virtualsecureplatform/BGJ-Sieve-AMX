#ifndef BGJ_FPLLL_BRIDGE_H
#define BGJ_FPLLL_BRIDGE_H

#include <NTL/mat_ZZ.h>

class Lattice_QP;

bool bgj_fplll_is_available();
int bgj_fplll_lll(NTL::Mat<NTL::ZZ> &basis, double delta, int verbose);
int bgj_fplll_lll_qp(Lattice_QP &basis, double delta, long ind_l, long ind_r, int verbose);
const char *bgj_fplll_status_string(int status);

#endif
