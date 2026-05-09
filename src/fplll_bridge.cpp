#include "../include/fplll_bridge.h"

#include "../include/lattice.h"

#include <sstream>
#include <string>
#include <vector>

#ifdef HAVE_FPLLL
#include <fplll/fplll.h>
#endif

bool bgj_fplll_is_available()
{
#ifdef HAVE_FPLLL
    return true;
#else
    return false;
#endif
}

const char *bgj_fplll_status_string(int status)
{
    switch (status) {
    case 0:
        return "success";
    case -1:
        return "fplll backend is not compiled in";
    case -2:
        return "failed to convert NTL basis to fplll basis";
    case -3:
        return "failed to convert fplll basis to NTL basis";
    case -4:
        return "fplll QP LLL supports only full local blocks";
    case -5:
        return "fplll QP GSO failed";
    default:
#ifdef HAVE_FPLLL
        static std::string status_text;
        status_text = fplll::get_red_status_str(status);
        return status_text.c_str();
#else
        return "unknown fplll bridge status";
#endif
    }
}

int bgj_fplll_lll(NTL::Mat<NTL::ZZ> &basis, double delta, int verbose)
{
#ifndef HAVE_FPLLL
    (void)basis;
    (void)delta;
    (void)verbose;
    return -1;
#else
    const long rows = basis.NumRows();
    const long cols = rows ? basis.NumCols() : 0;
    fplll::ZZ_mat<mpz_t> f_basis(rows, cols);

    for (long i = 0; i < rows; ++i) {
        for (long j = 0; j < cols; ++j) {
            std::ostringstream out;
            out << basis[i][j];
            if (!f_basis[i][j].set_str(out.str().c_str())) return -2;
        }
    }

    const int flags = verbose ? fplll::LLL_VERBOSE : fplll::LLL_DEFAULT;
    const int status = fplll::lll_reduction(f_basis,
                                            delta,
                                            fplll::LLL_DEF_ETA,
                                            fplll::LM_WRAPPER,
                                            fplll::FT_DEFAULT,
                                            0,
                                            flags);
    if (status != fplll::RED_SUCCESS) return status;

    NTL::Mat<NTL::ZZ> reduced;
    reduced.SetDims(rows, cols);
    for (long i = 0; i < rows; ++i) {
        for (long j = 0; j < cols; ++j) {
            std::ostringstream out;
            out << f_basis[i][j];
            std::istringstream in(out.str());
            if (!(in >> reduced[i][j])) return -3;
        }
    }
    basis = reduced;
    return 0;
#endif
}

int bgj_fplll_lll_qp(Lattice_QP &basis, double delta, long ind_l, long ind_r, int verbose)
{
#ifndef HAVE_FPLLL
    (void)basis;
    (void)delta;
    (void)ind_l;
    (void)ind_r;
    (void)verbose;
    return -1;
#else
    const long rows = basis.NumRows();
    const long cols = basis.NumCols();
    if (ind_l != 0 || ind_r != rows) return -4;

    fplll::ZZ_mat<double> f_basis(rows, cols);
    for (long i = 0; i < rows; ++i) {
        for (long j = 0; j < cols; ++j) {
            f_basis[i][j] = basis.get_b().hi[i][j];
        }
    }

    fplll::ZZ_mat<double> transform;
    fplll::ZZ_mat<double> transform_inv;
    transform.gen_identity(rows);

    fplll::MatGSO<fplll::Z_NR<double>, fplll::FP_NR<double>> gso(
        f_basis, transform, transform_inv, fplll::GSO_DEFAULT);
    if (!gso.update_gso()) return -5;

    const int flags = verbose ? fplll::LLL_VERBOSE : fplll::LLL_DEFAULT;
    fplll::LLLReduction<fplll::Z_NR<double>, fplll::FP_NR<double>> lll(
        gso, delta, fplll::LLL_DEF_ETA, flags);
    if (!lll.lll(ind_l, ind_l, ind_r, 0)) return lll.status;

    std::vector<double> transform_storage(rows * rows);
    std::vector<double *> transform_rows(rows);
    for (long i = 0; i < rows; ++i) {
        transform_rows[i] = transform_storage.data() + i * rows;
        for (long j = 0; j < rows; ++j) {
            transform_rows[i][j] = transform[i][j].get_d();
        }
    }
    basis.trans_by(transform_rows.data(), 0, rows);
    return 0;
#endif
}
