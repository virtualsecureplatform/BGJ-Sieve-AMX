#ifndef __SVP_H
#define __SVP_H

#include "lattice.h"
#include "config.h"

/** svp kernel based on 3-Sieve.
 *  \param msd maximal sieving dimension
 *  \param f   maximal depth of dim for free
 *  \param ni  maximal number of insertion
 *  \param ne  maximal number of extend_left after the main sieve
 *  \param ns  maximal number of sieving after the main sieve
 */
void __pump_red(Lattice_QP *L, long num_threads, double eta, long msd, long f, long ni, long ne, long ns);

/** svp kernel based on bgjf_epi8 & dual hash lift disabled.
 *  \param msd maximal sieving dimension
 *  \param f   maximal depth of dim for free
 *  \param ni  maximal number of insertion
 *  \param ne  maximal number of extend_left after the main sieve
 *  \param ns  maximal number of sieving after the main sieve
 */
void __pump_red_epi8(Lattice_QP *L, long num_threads, double eta, long msd, long f, long ni, long ne, long ns, long log_level, long shuffle_first = 1, long minsd = 40);

void __lsh_pump_red_epi8(Lattice_QP *L, long num_threads, double eta, double qratio, long msd, long f, long ni, long ne, long ns, long log_level, long shuffle_first = 1, long minsd = 40);

double __last_lsh_pump_epi8(Lattice_QP *L, long num_threads, double qratio, double ext_qratio, long msd, long ext_d, long log_level, long shuffle_first, long minsd);

#if defined(__AMX_INT8__)
void __pump_red_amx(Lattice_QP *L, long num_threads, double eta, long msd, long f, long ni, long ne, long ns, long log_level, long shuffle_first = 1, long minsd = 40);

void __lsh_pump_red_amx(Lattice_QP *L, long num_threads, double eta, double qratio, long msd, long f, long ni, long ne, long ns, long log_level, long shuffle_first = 1, long minsd = 40);

void __last_lsh_pump_amx(Lattice_QP *L, long num_threads, double qratio, double ext_qratio, long msd, long ext_d, long log_level, long shuffle_first, long minsd);
#endif

void __hkz_red(Lattice_QP *L, long num_threads);

long _red_60(Lattice_QP *L, long num_threads);
long _red_61(Lattice_QP *L, long num_threads);
long _red_62(Lattice_QP *L, long num_threads);
long _red_63(Lattice_QP *L, long num_threads);
long _red_64(Lattice_QP *L, long num_threads);
long _red_65(Lattice_QP *L, long num_threads);
long _red_66(Lattice_QP *L, long num_threads);
long _red_67(Lattice_QP *L, long num_threads);
long _red_68(Lattice_QP *L, long num_threads);
long _red_69(Lattice_QP *L, long num_threads);
long _red_70(Lattice_QP *L, long num_threads);
long _red_71(Lattice_QP *L, long num_threads);
long _red_72(Lattice_QP *L, long num_threads);
long _red_73(Lattice_QP *L, long num_threads);
long _red_74(Lattice_QP *L, long num_threads);
long _red_75(Lattice_QP *L, long num_threads);
long _red_76(Lattice_QP *L, long num_threads);
long _red_77(Lattice_QP *L, long num_threads);
long _red_78(Lattice_QP *L, long num_threads);
long _red_79(Lattice_QP *L, long num_threads);
long _red_80(Lattice_QP *L, long num_threads);
long _red_81(Lattice_QP *L, long num_threads);
long _red_82(Lattice_QP *L, long num_threads);
long _red_83(Lattice_QP *L, long num_threads);
long _red_84(Lattice_QP *L, long num_threads);
long _red_85(Lattice_QP *L, long num_threads);
long _red_86(Lattice_QP *L, long num_threads);
long _red_87(Lattice_QP *L, long num_threads);
long _red_88(Lattice_QP *L, long num_threads);
long _red_89(Lattice_QP *L, long num_threads);
long _red_90(Lattice_QP *L, long num_threads);
long _red_91(Lattice_QP *L, long num_threads);
long _red_92(Lattice_QP *L, long num_threads);
long _red_93(Lattice_QP *L, long num_threads);
long _red_94(Lattice_QP *L, long num_threads);
long _red_95(Lattice_QP *L, long num_threads);
long _red_96(Lattice_QP *L, long num_threads);
long _red_97(Lattice_QP *L, long num_threads);
long _red_98(Lattice_QP *L, long num_threads);
long _red_99(Lattice_QP *L, long num_threads);
long _red_100(Lattice_QP *L, long num_threads);
long _red_101(Lattice_QP *L, long num_threads);
long _red_102(Lattice_QP *L, long num_threads);
long _red_103(Lattice_QP *L, long num_threads);
long _red_104(Lattice_QP *L, long num_threads);
long _red_105(Lattice_QP *L, long num_threads);
long _red_106(Lattice_QP *L, long num_threads);
long _red_107(Lattice_QP *L, long num_threads);
long _red_108(Lattice_QP *L, long num_threads);
long _red_109(Lattice_QP *L, long num_threads);


long _pump_red_epi8_60(Lattice_QP *L, long num_threads);
long _pump_red_epi8_61(Lattice_QP *L, long num_threads);
long _pump_red_epi8_62(Lattice_QP *L, long num_threads);
long _pump_red_epi8_63(Lattice_QP *L, long num_threads);
long _pump_red_epi8_64(Lattice_QP *L, long num_threads);
long _pump_red_epi8_65(Lattice_QP *L, long num_threads);
long _pump_red_epi8_66(Lattice_QP *L, long num_threads);
long _pump_red_epi8_67(Lattice_QP *L, long num_threads);
long _pump_red_epi8_68(Lattice_QP *L, long num_threads);
long _pump_red_epi8_69(Lattice_QP *L, long num_threads);
long _pump_red_epi8_70(Lattice_QP *L, long num_threads);
long _pump_red_epi8_71(Lattice_QP *L, long num_threads);
long _pump_red_epi8_72(Lattice_QP *L, long num_threads);
long _pump_red_epi8_73(Lattice_QP *L, long num_threads);
long _pump_red_epi8_74(Lattice_QP *L, long num_threads);
long _pump_red_epi8_75(Lattice_QP *L, long num_threads);
long _pump_red_epi8_76(Lattice_QP *L, long num_threads);
long _pump_red_epi8_77(Lattice_QP *L, long num_threads);
long _pump_red_epi8_78(Lattice_QP *L, long num_threads);
long _pump_red_epi8_79(Lattice_QP *L, long num_threads);
long _pump_red_epi8_80(Lattice_QP *L, long num_threads);
long _pump_red_epi8_81(Lattice_QP *L, long num_threads);
long _pump_red_epi8_82(Lattice_QP *L, long num_threads);
long _pump_red_epi8_83(Lattice_QP *L, long num_threads);
long _pump_red_epi8_84(Lattice_QP *L, long num_threads);
long _pump_red_epi8_85(Lattice_QP *L, long num_threads);
long _pump_red_epi8_86(Lattice_QP *L, long num_threads);
long _pump_red_epi8_87(Lattice_QP *L, long num_threads);
long _pump_red_epi8_88(Lattice_QP *L, long num_threads);
long _pump_red_epi8_89(Lattice_QP *L, long num_threads);
long _pump_red_epi8_90(Lattice_QP *L, long num_threads);
long _pump_red_epi8_91(Lattice_QP *L, long num_threads);
long _pump_red_epi8_92(Lattice_QP *L, long num_threads);
long _pump_red_epi8_93(Lattice_QP *L, long num_threads);
long _pump_red_epi8_94(Lattice_QP *L, long num_threads);
long _pump_red_epi8_95(Lattice_QP *L, long num_threads);
long _pump_red_epi8_96(Lattice_QP *L, long num_threads);
#if COMPILE_POOL_EPI8_128
long _pump_red_epi8_97(Lattice_QP *L, long num_threads);
long _pump_red_epi8_98(Lattice_QP *L, long num_threads);
long _pump_red_epi8_99(Lattice_QP *L, long num_threads);
long _pump_red_epi8_100(Lattice_QP *L, long num_threads);
long _pump_red_epi8_101(Lattice_QP *L, long num_threads);
long _pump_red_epi8_102(Lattice_QP *L, long num_threads);
long _pump_red_epi8_103(Lattice_QP *L, long num_threads);
long _pump_red_epi8_104(Lattice_QP *L, long num_threads);
long _pump_red_epi8_105(Lattice_QP *L, long num_threads);
long _pump_red_epi8_106(Lattice_QP *L, long num_threads);
long _pump_red_epi8_107(Lattice_QP *L, long num_threads);
long _pump_red_epi8_108(Lattice_QP *L, long num_threads);
long _pump_red_epi8_109(Lattice_QP *L, long num_threads);
long _pump_red_epi8_110(Lattice_QP *L, long num_threads);
long _pump_red_epi8_111(Lattice_QP *L, long num_threads);
long _pump_red_epi8_112(Lattice_QP *L, long num_threads);
long _pump_red_epi8_113(Lattice_QP *L, long num_threads);
long _pump_red_epi8_114(Lattice_QP *L, long num_threads);
long _pump_red_epi8_115(Lattice_QP *L, long num_threads);
long _pump_red_epi8_116(Lattice_QP *L, long num_threads);
long _pump_red_epi8_117(Lattice_QP *L, long num_threads);
long _pump_red_epi8_118(Lattice_QP *L, long num_threads);
long _pump_red_epi8_119(Lattice_QP *L, long num_threads);
long _pump_red_epi8_120(Lattice_QP *L, long num_threads);
long _pump_red_epi8_121(Lattice_QP *L, long num_threads);
long _pump_red_epi8_122(Lattice_QP *L, long num_threads);
long _pump_red_epi8_123(Lattice_QP *L, long num_threads);
long _pump_red_epi8_124(Lattice_QP *L, long num_threads);
long _pump_red_epi8_125(Lattice_QP *L, long num_threads);
long _pump_red_epi8_126(Lattice_QP *L, long num_threads);
long _pump_red_epi8_127(Lattice_QP *L, long num_threads);
long _pump_red_epi8_128(Lattice_QP *L, long num_threads);
#endif
#if COMPILE_POOL_EPI8_160
long _pump_red_epi8_129(Lattice_QP *L, long num_threads);
long _pump_red_epi8_130(Lattice_QP *L, long num_threads);
long _pump_red_epi8_131(Lattice_QP *L, long num_threads);
long _pump_red_epi8_132(Lattice_QP *L, long num_threads);
long _pump_red_epi8_133(Lattice_QP *L, long num_threads);
long _pump_red_epi8_134(Lattice_QP *L, long num_threads);
long _pump_red_epi8_135(Lattice_QP *L, long num_threads);
long _pump_red_epi8_136(Lattice_QP *L, long num_threads);
long _pump_red_epi8_137(Lattice_QP *L, long num_threads);
long _pump_red_epi8_138(Lattice_QP *L, long num_threads);
long _pump_red_epi8_139(Lattice_QP *L, long num_threads);
long _pump_red_epi8_140(Lattice_QP *L, long num_threads);
long _pump_red_epi8_141(Lattice_QP *L, long num_threads);
long _pump_red_epi8_142(Lattice_QP *L, long num_threads);
long _pump_red_epi8_143(Lattice_QP *L, long num_threads);
long _pump_red_epi8_144(Lattice_QP *L, long num_threads);
long _pump_red_epi8_145(Lattice_QP *L, long num_threads);
long _pump_red_epi8_146(Lattice_QP *L, long num_threads);
long _pump_red_epi8_147(Lattice_QP *L, long num_threads);
long _pump_red_epi8_148(Lattice_QP *L, long num_threads);
long _pump_red_epi8_149(Lattice_QP *L, long num_threads);
long _pump_red_epi8_150(Lattice_QP *L, long num_threads);
long _pump_red_epi8_151(Lattice_QP *L, long num_threads);
long _pump_red_epi8_152(Lattice_QP *L, long num_threads);
long _pump_red_epi8_153(Lattice_QP *L, long num_threads);
long _pump_red_epi8_154(Lattice_QP *L, long num_threads);
long _pump_red_epi8_155(Lattice_QP *L, long num_threads);
long _pump_red_epi8_156(Lattice_QP *L, long num_threads);
long _pump_red_epi8_157(Lattice_QP *L, long num_threads);
long _pump_red_epi8_158(Lattice_QP *L, long num_threads);
long _pump_red_epi8_159(Lattice_QP *L, long num_threads);
long _pump_red_epi8_160(Lattice_QP *L, long num_threads);
#endif



#endif
