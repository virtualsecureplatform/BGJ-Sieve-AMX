#ifndef __CONFIG_H
#define __CONFIG_H

#ifndef __AVX512VNNI__
#error "avx512_vnni not found"
#endif

#ifndef BOOST_AMX_SIEVE
#define BOOST_AMX_SIEVE 1
#endif

#ifndef GSO_BLOCKSIZE
#define GSO_BLOCKSIZE 200
#endif

#ifndef JUMPING_STEP
#define JUMPING_STEP 8
#endif

#define COMPILE_POOL_EPI8_96 1
#define COMPILE_POOL_EPI8_128 1
#define COMPILE_POOL_EPI8_160 1
#define COMPILE_POOL_EPI8_192 1
#define COMPILE_POOL_EPI8_224 1

#define COMPILE_POOL_EPI8_MAX_DIM (96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32 + COMPILE_POOL_EPI8_192 * 32 + COMPILE_POOL_EPI8_224 * 32)

#define REJ_ENTRY128 1

// download it from github manually and put 
// it in dep/sparsepp before enabling this
#define USE_SPARSEPP 0
// ....do not enable it if USE_SPARSEPP is 1
#define UID_OP_INLINE 0

long check_avx512f();
long check_avx512vnni();
long check_amx();
long check_gso_blocksize();
long check_jumping_step();


#endif
