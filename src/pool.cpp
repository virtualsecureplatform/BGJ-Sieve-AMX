/** 
 * \warning: old codes based on float32, do not read it
*/
#include "../include/pool.h"
#include "../include/quad.h"

/* used to sort cvec. */
struct cvec_for_sort{
    __attribute__ ((packed)) long a[6];
};
inline bool cmp_cvec(cvec_for_sort a, cvec_for_sort b){
    return (((float *)&a.a[4])[0] < ((float *)&b.a[4])[0]);
}


//construction and distructions
Pool::Pool(){
    threadpool.resize(num_threads);
}
Pool::Pool(Lattice_QP *L){
    basis = L;
    if (basis->get_gso_status() == 0) basis->compute_gso_QP();
    threadpool.resize(num_threads);
}
Pool::~Pool(){
    clear_all();
}
void Pool::clear_all(){
    clear_pool();
    if (b_local){
        FREE_MAT((void **)b_local);
        b_local = NULL;
    }
    if (compress_pos) {
        delete[] compress_pos;
        compress_pos = NULL;
    }
    if (uid){
        delete uid;
        uid = NULL;
    }
}
void Pool::clear_pool(){
    if (cvec_store) {
        free(cvec_store);
        cvec_store = NULL;
    }
    if (vec_store) {
        free (vec_store);
        vec_store = NULL;
    }
    num_vec = 0;
    sorted_index = 0;
}

//setup
int Pool::set_num_threads(long n){
    num_threads = n;
    threadpool.resize(num_threads);
    omp_set_num_threads(num_threads);
    return 1;
}
int Pool::set_basis(Lattice_QP *L){
    clear_pool();
    basis = L;
    if (basis->get_gso_status() == 0) basis->compute_gso_QP();
    return 1;
}
int Pool::set_MSD(long msd){
    clear_pool();
    MSD = msd;
    vec_length = ((MSD-1)/16+1)*16;
    int_bias = ((MSD-1)/32+1)*16+16;
    vec_size = vec_length+int_bias; 
    Simhash_setup();
    return 1;
}
int Pool::set_max_pool_size(long N){
    clear_pool();
    max_pool_size = N;
    vec_store = (float *) calloc(4*vec_size*max_pool_size+512, 1);
    vec = (float *) ((((long)(vec_store)-1)/64+8)*64);
    cvec_store = (long *) calloc(8*cvec_size*max_pool_size+16, 1);
    cvec = (long *) ((((long)(cvec_store)-1)/16+1)*16);
    if ((cvec == NULL) || (vec == NULL)){
        clear_pool();
        return 0;
    }
    return 1;
}
int Pool::set_sieving_context(long l, long r){
    if (uid == NULL){
        uid = new UidHashTable();
        if (uid == NULL) return 0;
    }
    num_vec = 0;
    sorted_index = 0;
    index_l = l;
    index_r = r;
    CSD = r - l;
    compute_gh2();
    uid->reset_hash_function(CSD);
    update_b_local();
    return 1;
}
int Pool::compute_gh2(){
    double *B = basis->get_B().hi;
    double detn2 = 1.0;
    for (long i = index_l; i < index_r; i++){
        detn2 *= pow(B[i], 1.0/CSD);
    }
    gh2 = detn2 * pow(gh_coeff(CSD), 2.0);
    return 1;
}
void Pool::Simhash_setup(){
    if (compress_pos == NULL) {
        compress_pos = new uint32_t[256*6];
        if (compress_pos == NULL) return;
    }
    for (long i = 0; i < 256; i++){
        compress_pos[i*6+0] = i % MSD;
        compress_pos[i*6+1] = Uniform_long(MSD);
        compress_pos[i*6+2] = Uniform_long(MSD);
        compress_pos[i*6+3] = Uniform_long(MSD);
        compress_pos[i*6+4] = Uniform_long(MSD);
        compress_pos[i*6+5] = Uniform_long(MSD);
    }
    return;
}
void Pool::update_b_local(){
    if (b_local) {
        FREE_MAT((void **)b_local);
        b_local = NULL;
    }
    b_local = (float **)NEW_MAT(CSD, CSD, sizeof(float));
    if (b_local == NULL) return;
    double **miu = (basis->get_miu()).hi;
    double *B = (basis->get_B()).hi;
    for (long j = 0; j < CSD; j++){
        double x = sqrt(B[j+index_l]);
        for (long i = 0; i < CSD; i++){
            b_local[i][CSD-1-j] = miu[i+index_l][j+index_l] * x;
        }
    }
    return;
}

//pool operations
int Pool::gaussian_sampling(long N){
    if (N > max_pool_size){
        std::cerr << "[Warning] Pool::gaussian_sampling: N is larger than max_pool_size, aborted!\n";
        return 0;
    }
    if (N < num_vec){
        std::cerr << "[Info] Pool::gaussian_sampling: N is smaller than current pool size, aborted!\n";
        return 0;
    }
    bool success = true;
    #pragma omp parallel for 
    for (long thread = 0; thread < num_threads; thread++){
        DGS1d R(thread);
        for (long i = num_vec+thread; i < N; i+=num_threads){
            if (!success) continue;
            long count = 0;
            do {
                count++;
                if (count > 30){
                    std::cerr << "[Error] Pool::gaussian_sampling: sampling always get collision, aborted.\n";
                    success = false;
                    break;
                }
                gaussian_sampling(vec+i*vec_size, cvec+i*cvec_size, R);
            }while(!uid->insert_uid(*((uint64_t *)(&((vec+i*vec_size)[-4])))));
        }
    }
    if (success){
        num_vec = N;
        return 1;
    }
    return 0;
}
void Pool::gaussian_sampling(float *res, long *cres, DGS1d &R){
    set_zero(res-int_bias, vec_size);
    int coeff[CSD];
    double *B = (basis->get_B()).hi;
    double **miu = (basis->get_miu()).hi;
    double sigma2 = B[(index_l+index_r)/2];
    for (long i = CSD - 1; i >= 0; i--){
        coeff[i] = R.discrete_gaussian(res[i],sigma2/B[i+index_l]+0.1);
        for(long j = 0; j < i; j++){
            res[j] -= coeff[i]*miu[i+index_l][j+index_l];
        }
    }
    for (long i = 0; i < CSD; i++){
        ((short*)(&res[-int_bias]))[i] = coeff[CSD-1-i];
    }
    for (long i = CSD; i < MSD; i++){
        ((short*)(&res[-int_bias]))[i] = 0;
    }
    compute_vec(res);
    compute_cvec(res, cres);
}
int Pool::shrink(long N){
    if (N >= num_vec) {
        if (N > num_vec + 1000){
            std::cerr << "Pool::shrink: [Warning] N should be smaller than num_vec, shrink aborted.\n";
        }
        return 0;
    }
    sort_cvec();
    long tindex = N;
    for (long i = 0; i < N; i++){
        long *cptr = cvec + i * cvec_size;
        float *ptr = (float *) cptr[5];
        if ((ptr-vec)<vec_size*N)continue;
        float *dst;
        long *cdst;
        do{ 
            if (tindex >= num_vec){
                std::cerr << "[Error] Pool::shrink: shrink failed! something must be wrong, pool clobbered!\n";
                return 0;
            }
            cdst = cvec + tindex * cvec_size;
            dst = (float *)(cdst[5]);
            tindex++;
        } while((dst -vec)>=vec_size*N);
        ((float **)cptr)[5] = dst;
        copy(dst-int_bias, ptr-int_bias,vec_size);
    }
    num_vec = N;
    sorted_index = N;
    return 1;
}
/* something must be wrong with it, the length 
 * of the vectors after several lifting will fail,
 * but it seems to work well in 3-Sieve. maybe 
 * it's caused by floating point error?  --2022.9.18
 * to fix
 */
int Pool::extend_left(){
    if (index_l == 0){
        std::cerr << "[Warning] Pool::extend_left: index_l = 0, cannot extend_left, aborted.\n";
        return 0;
    }
    index_l--;
    CSD++;
    update_b_local();
    compute_gh2();
    uid->reset_hash_function(CSD);

    #pragma omp parallel for 
    for (long i = 0; i < num_vec; i++){
        long *cres = cvec + i * cvec_size;
        float *res = (float *)(cres[5]);
        short *x = (short *)(&res[-int_bias]);
        float y = 0.0;
        for (long j = 0; j < CSD-1; j++){
            y += x[CSD-2-j]*b_local[j+1][CSD-1];
        }
        short y_ = round(y/b_local[0][CSD-1]);
        y = y - y_*b_local[0][CSD-1];
        x[CSD-1] = -y_;
        res[CSD-1] = y;
        compute_Simhash(res);
        compute_uid(res);
        res[-1] = norm(res, vec_length);
        *((uint64_t *)(&cres[0])) = *((uint64_t *)(&res[-16]));
        *((uint64_t *)(&cres[1])) = *((uint64_t *)(&res[-14]));
        *((uint64_t *)(&cres[2])) = *((uint64_t *)(&res[-12]));
        *((uint64_t *)(&cres[3])) = *((uint64_t *)(&res[-10]));
        *((float *)(&cres[4])) = res[-1];
        if (!uid->insert_uid(*((uint64_t *)(&(res[-4]))))){
            std::cerr << "[Error] Pool::extend_left: uid collision while extend left, something must wrong, aborted.\n";
            //int *a = 0x0;
            //std::cout << a[0] << std::endl; 
        }
    }
    sorted_index = 0;
    return 1;
}
int Pool::shrink_left(){
    #pragma omp parallel for
    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        short *x = (short *) (ptr - int_bias);
        x[CSD - 1] = 0;
    }
    index_l++;
    CSD--;
    compute_gh2();
    uid->reset_hash_function(CSD);
    update_b_local();
    sorted_index = 0;

    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        compute_uid(ptr);
        if (uid->insert_uid(*((uint64_t *)(&(ptr[-4]))))) continue;
        if (num_vec > i + 1){
            num_vec--;
            copy(ptr-int_bias, vec + num_vec * vec_size -int_bias, int_bias - 16);
        }else{
            num_vec--;
            break;
        }
        i--;
    }

    #pragma omp parallel for
    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        long *cptr = cvec + i * cvec_size;
        compute_vec(ptr);
        compute_cvec(ptr, cptr);
    }
    return 1;
}
int Pool::insert(long index, double delta){
    long coeff_size = (int_bias-16)*2;
    long coeff_block = coeff_size/32;
    if (index > index_l){
        std::cerr << "[Warning] Pool::insert: insertion index larger than index_l, aborted\n";
        return 0;
    }
    if (index < 0){
        std::cerr<< "[Warning] Pool::insert: negetive insertion index, aborted\n";
        return 0;
    }
    long min_index = -1;
    long min_place = -1;
    pthread_spinlock_t min_lock;
    pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);


    long ID = index_l - index;
    long FD = index_r - index;
    float min_score = pow(delta, ID);
    float min_ratio = 0.995;
    long ID16 = ((ID+15)/16)*16;
    float *Bi = new float[ID+1];
    float *Bis = new float[ID+1];
    float *Ci = new float[ID+1];
    float **b_insert = (float **) NEW_MAT(FD, ID16, sizeof(float));
    double **b = basis->get_b().hi;
    double **miu = basis->get_miu().hi;
    double *B = basis->get_B().hi;
    for (long j = 0; j < ID; j++){
        Bis[j] = sqrt(B[index+j]);
        for (long i = 0; i < FD; i++){
            b_insert[i][j] = miu[i+index][j+index]*Bis[j];
        }
        Bis[j] = 1.0/Bis[j];
    }
    for (long i = 0; i <= ID; i++){
        Bi[i] = 1.0/B[index + i];
    }
    for (long i = 0; i <= ID; i++){
        Ci[i] = Bi[i]*pow(delta, i);
    }

    //find the best insertion
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++){
        __attribute__ ((aligned (64))) float tmp_store[vec_size+16];
        float *tmp = &tmp_store[16];        
        for (long i = thread; i < num_vec; i+=num_threads){
            long *cptr = cvec + i * cvec_size;
            float *ptr = (float *) cptr[5];
            short *x = (short *)(ptr - int_bias);
            set_zero(tmp, vec_size);
            for (long j = 0; j < CSD; j++){
                red(tmp, b_insert[j+ID], -x[CSD - 1 - j], ID16);
            }
            for (long j = ID - 1; j >= 0; j--){
                red(tmp, b_insert[j], round(Bis[j]*tmp[j]), ID16); 
            }
            float norm1 = ptr[-1];
            for (long ind = ID; ind >= 0; ind--){
                if (norm1 * Bi[ind] < 0.995){
                    if (norm1 * Ci[ind] < min_score){
                        pthread_spin_lock(&min_lock);
                        if (norm1 * Ci[ind] < min_score){
                            bool hasone = false;
                            for (long j = 0; j < CSD; j++){
                                if (x[j] == 1) hasone = true;
                                if (x[j] == -1) hasone = true;
                            }
                            if (hasone){
                                min_score = norm1 * Ci[ind];
                                min_index = i;
                                min_place = ind+index;
                                min_ratio = norm1 * Bi[ind];
                                //std::cout << "new minimal score:" << min_score << std::endl;
                            }
                        }
                        pthread_spin_unlock(&min_lock);
                    }
                }
                norm1 += tmp[ind-1] * tmp[ind-1];
            }
        }
    }
    if (min_index == -1){
        //std::cerr << "insertion failed, aborted.\n";
        shrink_left();
        FREE_MAT((void **)b_insert);
        delete[] Bi;
        delete[] Bis;
        delete[] Ci;
        return 0;
    }else{
        //printf("seiving context = [%ld, %ld] ",index_l, index_r);
        //std::cerr << min_index << "-th vec insert to index " << min_place << ", score = " << sqrt(min_score) << ", ratio = " << sqrt(min_ratio) << std::endl;
    }
    __attribute__ ((aligned (64))) float tmp_store[vec_size];
    float *tmp = &tmp_store[0];
    long *cptr = cvec + min_index * cvec_size;
    float *ptr = (float *)cptr[5];
    short *x = (short *)(ptr - int_bias);

    VEC_QP v = NEW_VEC_QP(basis->NumCols());
    MAT_QP b_QP = basis->get_b();
    long rm_index;
    for (long i = 0; i < CSD; i++){
        if (abs(x[CSD - 1 - i]) == 1){
            rm_index = CSD - 1 - i;
            break;
        }
        if (i == (CSD - 1)){
            std::cerr << "[Error] Pool::insert: something must be done while insertion!\n";
            return 0;    
        }
    }
    set_zero(tmp, vec_size);
    for (long j = 0; j < CSD; j++){
        NTL::quad_float x1(-x[CSD-1-j]);
        red(v.hi, v.lo, b_QP.hi[j+index_l], b_QP.lo[j+index_l], x1, basis->NumCols());
        red(tmp, b_insert[j+ID], -x[CSD - 1 - j], ID16);
    }
    for (long j = ID - 1; j >= min_place - index; j--){
        double y = round(Bis[j]*tmp[j]);
        NTL::quad_float y1(y);
        red(v.hi, v.lo, b_QP.hi[j + index_l - ID], b_QP.lo[j + index_l - ID], y1, basis->NumCols());
        red(tmp, b_insert[j], y, ID16);
    }

    //compute the new coeff
    #pragma omp parallel for
    for (long i = 0; i < num_vec; i++){
        if (i == min_index) continue;
        long *cptr1 = cvec + i * cvec_size;
        float *ptr1 = (float *)cptr1[5];
        short *x1 = (short *)(ptr1-int_bias);
        if (x[rm_index] > 0){
            red(x1, x, x1[rm_index], coeff_size);
        }else{
            red(x1, x, -x1[rm_index], coeff_size);
        }
        for (long j = rm_index; j < CSD - 1; j++){
            x1[j] = x1[j+1];
        }
        x1[CSD-1] = 0;
    }
    red(x,x,1,coeff_size);

    index_l++;
    CSD--;    
    for (long i = CSD - rm_index+index_l-1; i > min_place; i--){
        copy(b_QP.hi[i], b_QP.lo[i], b_QP.hi[i-1], b_QP.lo[i-1], basis->NumCols());
    }
    copy(b_QP.hi[min_place], b_QP.lo[min_place], v.hi, v.lo, basis->NumCols());
    basis->compute_gso_QP();
    basis->size_reduce(min_place);

    compute_gh2();
    uid->reset_hash_function(CSD);
    update_b_local();

    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        compute_uid(ptr);
        if (uid->insert_uid(*((uint64_t *)(&(ptr[-4]))))) continue;
        if (num_vec > i + 1){
            num_vec--;
            copy((ptr-int_bias), (vec + num_vec * vec_size -int_bias), int_bias - 16);
        }else{
            num_vec--;
            break;
        }
        i--;
    }
    #pragma omp parallel for
    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        long *cptr = cvec + i * cvec_size;
        compute_vec(ptr);
        compute_cvec(ptr, cptr);
    }
    sorted_index = 0;
    FREE_MAT((void **)b_insert);
    delete[] Bi;
    delete[] Bis;
    delete[] Ci;
    return 1;
}
int Pool::show_min_lift(long index){
    long coeff_size = (int_bias-16)*2;
    long coeff_block = coeff_size/32;
    if (index > index_l){
        std::cerr << "[Warning] Pool::show_min_lift: insertion index larger than index_l, aborted\n";
        return 0;
    }
    if (index < 0){
        std::cerr<< "[Warning] Pool::show_min_lift: negetive insertion index, aborted\n";
        return 0;
    }
    double *B = basis->get_B().hi;
    MAT_QP b_QP = basis->get_b();
    double **b = basis->get_b().hi;
    double **miu = basis->get_miu().hi;
    long min_index= -1;
    float min_norm = B[index]*0.995;
    pthread_spinlock_t min_lock;
    pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);


    long ID = index_l - index;
    long FD = index_r - index;
    long ID16 = ((ID+15)/16)*16;
    float **b_insert = new float*[FD];
    float *Bi = new float[ID];
    float *b_insert_store = (float *) calloc(FD*ID16*4+64, 1);
    float *b_insert_start = (float *) ((((long)(b_insert_store)-1)/64+1)*64);
    for (long i = 0; i < FD; i++){
        b_insert[i] = b_insert_start + i * ID16;
    }
    for (long j = 0; j < ID; j++){
        Bi[j] = sqrt(B[index+j]);
        for (long i = 0; i < FD; i++){
            b_insert[i][j] = miu[i+index][j+index]*Bi[j];
        }
        Bi[j] = 1.0/Bi[j];
    }
    

    //find the best insertion
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++){
        __attribute__ ((aligned (64))) float tmp_store[vec_size];
        float *tmp = &tmp_store[0];        
        for (long i = thread; i < num_vec; i+=num_threads){
            long *cptr = cvec + i * cvec_size;
            float *ptr = (float *) cptr[5];
            short *x = (short *)(ptr - int_bias);
            set_zero(tmp, vec_size);
            for (long j = 0; j < CSD; j++){
                red(tmp, b_insert[j+ID], -x[CSD - 1 - j], ID16);
            }
            for (long j = ID - 1; j >= 0; j--){
                red(tmp, b_insert[j], round(Bi[j]*tmp[j]), ID16); 
            }
            float norm1 = norm(tmp, ID16) + ptr[-1];
            if (norm1 < min_norm){
                pthread_spin_lock(&min_lock);
                if (norm1 < min_norm){
                    bool hasone = false;
                    for (long j = 0; j < CSD; j++){
                        if (x[j] == 1) hasone = true;
                        if (x[j] == -1) hasone = true;
                    }
                    if (hasone){
                        min_norm = norm1;
                        min_index = i;
                    }
                }
                pthread_spin_unlock(&min_lock);
            }
        }
    }
    if (min_index == -1){
        return 1;
    }
    __attribute__ ((aligned (64))) float tmp_store[vec_size];
    float *tmp = &tmp_store[0];
    long *cptr = cvec + min_index * cvec_size;
    float *ptr = (float *)cptr[5];
    short *x = (short *)(ptr - int_bias);
    VEC_QP v = NEW_VEC_QP(basis->NumCols());
    set_zero(tmp, vec_size);
    for (long j = 0; j < CSD; j++){
        NTL::quad_float x1 = NTL::to_quad_float(-x[CSD - 1 - j]);
        red(v.hi, v.lo, b_QP.hi[j+index_l], b_QP.lo[j + index_l],x1, basis->NumCols());
        red(tmp, b_insert[j+ID], -x[CSD - 1 - j], ID16);
    }
    for (long j = ID - 1; j >= 0; j--){
        double y = round(Bi[j]*tmp[j]);
        NTL::quad_float y1 = NTL::to_quad_float(y);
        red(v.hi, v.lo, b_QP.hi[j + index_l - ID], b_QP.lo[j + index_l - ID],y1, basis->NumCols());
        red(tmp, b_insert[j], y, ID16);
    }
    std::cout << "\nlength = " << sqrt(dot(v.hi,v.hi, basis->NumCols())) << ", vec =";
    std::cout << "[";
    for (long i = 0; i < basis->NumCols()-1; i++){
        std::cout << v.hi[i] << " ";
    }
    std::cout << v.hi[basis->NumCols() -1] << "]\n";
    delete[] b_insert;
    delete[] Bi;
    free(b_insert_store);
    return 1;
}
int Pool::tail_LLL(double delta, long n){
    long left = index_r - n;
    Lattice_QP b_tail;
    b_tail.set_size(n, n);
    double **bt = b_tail.get_b().hi;

    double *Bs = (double *) NEW_VEC(n, sizeof(double));
    double **miu = basis->get_miu().hi;
    MAT_QP b = basis->get_b();
    MAT_QP miu_QP = basis->get_miu();
    double *B = basis->get_B().hi;
    
    for (long j = 0; j < n; j++){
        Bs[j] = sqrt(B[j+left]);
        for (long i = 0; i < n; i++){
            bt[i][j] = miu[i + left][j + left] * Bs[j];
        }
    }
    b_tail.LLL_QP(delta);
    double **A = (double **) NEW_MAT(n, n, sizeof(double));
    for (long i = 0; i < n; i++){
        for (long j = n-1; j >= 0; j--){
            double x = bt[i][j]/Bs[j];
            x = x - round(x);
            if (fabs(x) > 0.1) {
                std::cerr << "[Error] Pool::tail_LLL: wrong coeff in tail_LLL, aborted\n";
                FREE_VEC((void *)Bs);
                FREE_MAT((void **)A);
                return 0;
            }
            A[i][j] = round(bt[i][j]/Bs[j]);
            for (long k = 0; k < n; k++){
                bt[i][k] = bt[i][k] - A[i][j] * (miu[left+j][left+k] * Bs[k]);
            }
        }
    }

    long **Ai = (long **) NEW_MAT(n, n, sizeof(long));
    int_inv(Ai, A, n);

    basis->trans_by(A, left, left + n);
    basis->compute_gso_QP(left);
    basis->set_gso_status(GSO_COMPUTED_QP);
    long **BB = (long **) NEW_MAT(n, CSD-n, sizeof(long));
    for (long indd = left; indd < left + n; indd++){
        long x_ind = indd - left;
        x_ind = n - 1 - x_ind;
        for (long i = left-1; i >= 0; i --){
            NTL::quad_float q = NTL::to_quad_float(round(miu[indd][i]));
            if (fabs(q.hi) > 0.7){
                red(miu_QP.hi[indd], miu_QP.lo[indd], miu_QP.hi[i], miu_QP.lo[i], q, i + 1);
                red(b.hi[indd], b.lo[indd], b.hi[i], b.lo[i], q, basis->NumCols());
            }
            if (i >= index_l){
                BB[x_ind][left - 1 - i] = round(q.hi);
            }
        }
    }

    update_b_local();
    uid->reset_hash_function(CSD);

    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++){
        short *x_tmp = new short[n];
        for (long i = thread; i < num_vec; i += num_threads){
            long *cptr1 = cvec + i * cvec_size;
            float *ptr1 = (float *)cptr1[5];
            short *x1 = (short *)(ptr1-int_bias);
            for (long j = 0; j < n; j++){
                x_tmp[j] = 0;
            }
            for (long j = 0; j < n; j++){
                for (long k = 0; k < n; k++){
                    x_tmp[k] += Ai[j][k] * x1[n-1-j];
                }
            }
            for (long j = 0; j < n; j++){
                x1[n-1-j] = x_tmp[j];
            }
            for (long j = 0; j < n; j++){
                for (long k = n; k < CSD; k++){
                    x1[k] += BB[j][k-n] * x1[j];
                }
            }
            compute_vec(ptr1);
            compute_cvec(ptr1, cptr1);
            if (!uid->insert_uid(*((uint64_t *)(&(ptr1[-4]))))){
                std::cerr << "[Error] Pool::tail_LLL: something must be wrong in tail_LLL uid insertion, aborted\n";
            }
        }
        delete[] x_tmp; 
    }

    FREE_MAT((void **)BB);
    FREE_MAT((void **)Ai);
    FREE_MAT((void **)A);
    FREE_VEC((void *)Bs);
    return 1;
}
int Pool::sort_cvec(){
    cvec_for_sort *start = (cvec_for_sort *)cvec;
    cvec_for_sort *middle = (cvec_for_sort *)(cvec + sorted_index*cvec_size);
    cvec_for_sort *end = (cvec_for_sort *)(cvec + num_vec*cvec_size);
    if (sorted_index == num_vec) return 1;
    if (sorted_index > num_vec/4){
        parallel_algorithms::sort(middle, end, cmp_cvec,threadpool);
        cvec_for_sort *tmp = new cvec_for_sort[num_vec];
        parallel_algorithms::merge(start, middle, middle, end, tmp, cmp_cvec, threadpool);
        parallel_algorithms::copy(tmp, tmp+num_vec, start, threadpool);
        delete[] tmp;
    }else{
        parallel_algorithms::sort(start, end, cmp_cvec, threadpool);
    }
    sorted_index = num_vec;
    return 1;   
}
int Pool::sieve_is_over(double saturation_radius, double saturation_ratio, bool show_details){
    float goal = gh2 * saturation_radius;
    long goal_num = saturation_ratio * 0.5 * pow(saturation_radius, CSD/2.0);
    long num_in_sorted_part = 0;            //this name is a little confusing
    long up = sorted_index;
    long down = 0;
    while (up > down + 1){
        long mid = (up+down+1)/2;
        if (((float *)(cvec+mid*cvec_size+4))[0] < goal){
            down = mid;
        }else{
            up = mid;
        }
    }
    num_in_sorted_part = up;
    if (!show_details){
        if (num_in_sorted_part >= goal_num) return 1;
        for (long i = sorted_index; i < num_vec; i++){
            if (i % 100 == 0){
                if (num_in_sorted_part + num_vec-i < goal_num) return 0;
                if (num_in_sorted_part >= goal_num) return 1;
            }
            if (((float *)(cvec+i*cvec_size+4))[0] < goal) num_in_sorted_part++;
        }
        if (num_in_sorted_part >= goal_num) return 1;
    }else{
        for (long i = sorted_index; i < num_vec; i++){
            if (((float *)(cvec+i*cvec_size+4))[0] < goal) num_in_sorted_part++;
        }
        printf("\rsieving on [%ld, %ld], %ld/%ld vecs found", index_l, index_r, num_in_sorted_part, goal_num);
        std::cout << std::flush;
        if (num_in_sorted_part >= goal_num) return 1;
    }
    return 0;
}
int Pool::store(const char *file_name){
    std::ofstream hout (file_name, std::ios::out); 
    hout << MSD << std::endl;
    hout << index_l << std::endl;
    hout << index_r << std::endl;
    hout << max_pool_size << std::endl;
    hout << "[";
    for (long i = 0; i < num_vec; i++){
        long *cptr = cvec + i * cvec_size;
        float *ptr = (float *)cptr[5];
        ptr -= int_bias;
        short *x = (short *)ptr;
        hout << "[";
        for (long j = CSD-1; j > 0; j--){
            hout << x[j] << " ";
        }
        hout << x[0]<<"]\n";
    }
    hout << "]";
    return 1;
}
int Pool::load(const char *file_name){
    clear_pool();
    std::ifstream data (file_name, std::ios::in);
    data >> MSD;
    data >> index_l;
    data >> index_r;
    data >> max_pool_size;
    CSD = index_r-index_l;
    update_b_local();
    compute_gh2();
    if (uid == NULL){
        uid = new UidHashTable();
        if (uid == NULL) return 0;
    }
    uid->reset_hash_function(CSD);
    set_MSD(MSD);
    set_max_pool_size(max_pool_size);
    NTL::Mat<short> coeff;
    data >> coeff;
    num_vec = coeff.NumRows();
    for (long i = 0; i < num_vec; i++){
        long *cptr = cvec + i * cvec_size;
        float *ptr = vec + i * vec_size;
        short *x = (short *)(ptr-int_bias);
        for (long j = 0; j < CSD; j++){
            x[j] = coeff[i][CSD-1-j];
        }
        compute_vec(ptr);
        compute_cvec(ptr, cptr);
        uid->insert_uid(*((uint64_t *)(&ptr[-4])));
    }
    return 1;
}
int Pool::store_vec(const char *file_name){
    sort_cvec();
    std::ofstream hout (file_name, std::ios::out);
    hout << num_vec << std::endl;
    hout << "[" << index_l << ", " << index_r << "]\n" << std::endl;
    hout << (*basis) << std::endl;
    long m = basis->NumCols();
    long m16 = ((m + 15)/16)*16;
    float **b_int = (float **)NEW_MAT(CSD, m16, sizeof(float));
    double **b = (basis->get_b()).hi;
    for (long i = 0; i < CSD; i++){
        for (long j = 0; j < basis->NumCols(); j++){
            b_int[i][j] = b[i+index_l][j];
        }
    }
    float *tmp = (float *)NEW_VEC(m16, sizeof(float));
    for (long i = 0; i < num_vec; i++){
        long *cptr = cvec + i * cvec_size;
        float *ptr = (float *)cptr[5];
        ptr -= int_bias;
        short *x = (short *)ptr;
        set_zero(tmp, m16);
        for (long j = 0; j < CSD; j++){
            red(tmp, b_int[j], -x[CSD - 1 - j], m16);
        }
        hout << "[";
        for (long j = 0; j < m-1; j++){
            hout << tmp[j]<< " ";
        }
        hout << tmp[m-1]<< "]\n";
    }
    FREE_VEC(tmp);
    FREE_MAT((void **)b_int);
    return 1;
}
double Pool::pot(){
    double *B = basis->get_B().hi;
    double pot = 0.0;
    for (long i = 0; i < basis->NumRows(); i++){
        pot += (basis->NumRows()-i) * log2(B[i]);
    }
    return pot;
}
bool Pool::check_pool_status(){
    bool ok = true;
    for (long i = 0; i < num_vec; i++){
        long *cptr = cvec + cvec_size * i;
        float *ptr = (float *)cptr[5];
        uint64_t u = *((uint64_t *)(&ptr[-4]));
        bool s0 = !(*((uint64_t *)(&cptr[0])) == *((uint64_t *)(&ptr[-16])));
        bool s1 = !(*((uint64_t *)(&cptr[1])) == *((uint64_t *)(&ptr[-14])));
        bool s2 = !(*((uint64_t *)(&cptr[2])) == *((uint64_t *)(&ptr[-12])));
        bool s3 = !(*((uint64_t *)(&cptr[3])) == *((uint64_t *)(&ptr[-10])));
        if (s0 || s1 || s2 || s3){
            ok = false;
            std::cerr << "the "<< i << "-th vector, Simhash do not match!\n";
        }
        uint64_t sim_old[4];
        compute_vec(ptr);
        s0 = !(*((uint64_t *)(&cptr[0])) == *((uint64_t *)(&ptr[-16])));
        s1 = !(*((uint64_t *)(&cptr[1])) == *((uint64_t *)(&ptr[-14])));
        s2 = !(*((uint64_t *)(&cptr[2])) == *((uint64_t *)(&ptr[-12])));
        s3 = !(*((uint64_t *)(&cptr[3])) == *((uint64_t *)(&ptr[-10])));
        if (s0 || s1 || s2 || s3){
            ok = false;
            std::cerr << "the "<< i << "-th vector, cvec Simhash wrong!\n";
            std::cerr << "old_simhash = [";
            std::cerr << std::hex << *((uint64_t *)(&cptr[0])) << ", ";
            std::cerr << std::hex << *((uint64_t *)(&cptr[1])) << ", ";
            std::cerr << std::hex << *((uint64_t *)(&cptr[2])) << ", ";
            std::cerr << std::hex << *((uint64_t *)(&cptr[3]));
            std::cerr <<"], new simhash = [";
            std::cerr << std::hex << *((uint64_t *)(&ptr[-16])) << ", ";
            std::cerr << std::hex << *((uint64_t *)(&ptr[-14])) << ", ";
            std::cerr << std::hex << *((uint64_t *)(&ptr[-12])) << ", ";
            std::cerr << std::hex << *((uint64_t *)(&ptr[-10]));
            std::cerr <<"]\n";
            std::cerr << std::dec;
        }
        if (u != *((uint64_t *)(&ptr[-4]))){
            ok = false;
            std::cerr << "the "<< i << "-th vector, uid wrong!\n";
        }
        if (!uid->check_uid(*((uint64_t *)(&ptr[-4])))){
            ok = false;
            std::cerr << "the "<< i << "-th vector, uid not in hashtable!\n";
        }
    }
    if (uid->size() > num_vec + 1) {ok = false;}
    if (!ok) std::cerr << "there is " << uid->size() << " elements in hashtable, "<< num_vec << " vec in the pool\n";
    return ok;
}
void Pool::check_dim_lose(){
    for (long ind = 0; ind < CSD; ind++){
        long count = 0;
        for (long i = 0; i < num_vec; i++){
            float *ptr = vec + i * vec_size;
            ptr -= int_bias;
            if (((short *)ptr)[ind] != 0) {
                count ++;
            }
        }
        if (count < 0.05 * num_vec) {
            if (count == 0) {
                std::cerr << "[Warning] the " << (CSD - 1 - ind) << "-th vec is definitely lose.\n";
                continue;
            }
            std::cerr << "[Warning] the " << (CSD - 1 - ind) << "-th vec may be lose, only appears in "<< count << "/" << num_vec << "vectors.\n";
        }
    } 
}

coeff_buffer::coeff_buffer(long coeffsize, long maxsize){
    max_size = maxsize;
    coeff_size = coeffsize;
    buffer_store = (short *)malloc(2 * max_size * coeff_size + 64);
    buffer = (short *) ((((long)(buffer_store)-1)/64+1)*64);
}
coeff_buffer::~coeff_buffer(){
    if (buffer_store != NULL){
        free(buffer_store);
        buffer_store = NULL;
    }
}
void coeff_buffer::buffer_setup(long coeffsize, long maxsize){
    if (buffer_store == NULL){
        max_size = maxsize;
        coeff_size = coeffsize;
        buffer_store = (short *)malloc(2 * max_size * coeff_size + 64);
        buffer = (short *) ((((long)(buffer_store)-1)/64+1)*64);
    }else{
        printf("buffer already set up, nothing done!\n");
    }
}
