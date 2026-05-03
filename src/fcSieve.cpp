/** 
 * \warning: old codes based on float32, do not read it
*/
#include "../include/pool.h"
#include "../include/vec.h"
#include <sys/time.h>

/* warning! to change center_hwt in Tfc_Sieve_nsh, 
 * we should change the code in the bucketing stage also
 * search "CHANGE_WITH_CENTER_HWT" to find the place to change
 */
#define XPC_FCS_CENTER_HWT 2
#define NPB_VEC_SIZE 5


class N_ptr_bucket{
    public:
        N_ptr_bucket(){};
        N_ptr_bucket(long dim, long alloc_size = 8);
        ~N_ptr_bucket(){clear();};

        long dim;
        long alloc_size;
        
        long *sparse_ind = NULL;
        float ***ptr = NULL;
        long *num = NULL;
        long *max_size = NULL;

        void alloc();
        void clear();
        void set_params(long dim, long alloc_size = 8);
        void add(float *vec, uint64_t sh0, uint64_t sh1, uint64_t sh2, uint64_t sh3, long ind);
};

int Pool::fc_Sieve(fc_Sieve_params params){
    // this is a first version, we will not warning on stucking.
    long coeff_size = (int_bias-16)*2;
    long coeff_block = coeff_size/32;
    if (params.show_details){
        std::cerr << "\n";
        std::cerr << "begin fcSieve on context ["<< index_l<<", "<<index_r << "], gh = "<< sqrt(gh2);
        std::cerr << ", pool size = "<< num_vec<<", "<<num_threads<<" threads will be used\n";
    }
    sort_cvec();
    
    long count = -1;    //record the number of epoches
    //begin the main loops
    long ttt = 0;       //record the number of sols we found, will be removed in the final version
    while (!sieve_is_over(params.saturation_radius, params.saturation_ratio)){
        //set the goal_norm of this epoch
        count++;
        long goal_index = (long)(params.improve_ratio*num_vec);
        float goal_norm = ((float *)(cvec+goal_index*cvec_size+4))[0];

        //prepare the buffer
        coeff_buffer local_buffer[num_threads];
        coeff_buffer main_buffer(coeff_size, num_vec);
        for (long i = 0; i < num_threads; i++){
            local_buffer[i].buffer_setup(coeff_size, num_vec/num_threads);
        }

        //collect solutions
        bool rel_collection_stop = false;
        double bucketing_time = 0.0;
        double search_time = 0.0;
        long ccount = 0;
        long cfound = 0;
        long cavg_bucket_size = 0;
        long calready_in = 0;
        long ctest_simhash = 0;
        long ctest_dot = 0;
        pthread_spinlock_t comb_lock = 1;
        while(!rel_collection_stop){
            ccount++;
            //get several buckets, number of buckets should be divided by numthreads
            //first prepare for buckets
            struct timeval startb, endb;    //record time
            gettimeofday(&startb, NULL);
            N_ptr_bucket bucket_list[params.batch_size];
            N_ptr_bucket localbucket_list[params.batch_size*num_threads];
            for (long i = 0; i < params.batch_size*num_threads; i++) {
                localbucket_list[i].set_params(params.center_dim, 8);
            }
            for (long i = 0; i < params.batch_size; i++){
                bucket_list[i].dim = params.center_dim;
                bucket_list[i].sparse_ind = (long *) NEW_VEC(XPC_FCS_CENTER_HWT*params.center_dim, sizeof(long));
                bucket_list[i].ptr = new float**[1L<<params.center_dim];
                bucket_list[i].num = new long[1L<<params.center_dim];
            }
            for (long i = 0; i < params.batch_size; i++) {
                long num_index_per_center = params.center_dim*XPC_FCS_CENTER_HWT;
                if (num_index_per_center*2 <= CSD){
                    long _22827xyz = 0;
                    do {
                        long x = Uniform_long(CSD);
                        bool ain = false;
                        for (long j = 0; j < _22827xyz; j++){
                            if (x == bucket_list[i].sparse_ind[j]){
                                ain = true;
                                break;
                            }
                        }
                        if (!ain) {
                            bucket_list[i].sparse_ind[_22827xyz] = x;
                            _22827xyz++;
                        }
                    } while (_22827xyz < num_index_per_center);
                }else{
                    long *not_in_list = new long[CSD - num_index_per_center];
                    long _22827xyz = 0;
                    do {
                        long x = Uniform_long(CSD);
                        bool ain = false;
                        for (long j = 0; j < _22827xyz; j++){
                            if (x == not_in_list[j]){
                                ain = true;
                                break;
                            }
                        }
                        if (!ain) {
                            not_in_list[_22827xyz] = x;
                            _22827xyz++;
                        }
                    }while(_22827xyz < CSD - num_index_per_center);
                    _22827xyz = 0;
                    for (long j = 0; j < CSD; j++){
                        bool ain = false;
                        for (long k = 0; k < CSD - num_index_per_center; k++){
                            if (j == not_in_list[k]){
                                ain = true;
                                break;
                            }
                        }
                        if (!ain){
                            bucket_list[i].sparse_ind[_22827xyz] = j;
                            _22827xyz++;
                        }
                    }
                    for (long j = 0; j < num_index_per_center; j++){
                        long swapind = Uniform_long(num_index_per_center);
                        long tmp = bucket_list[i].sparse_ind[j];
                        bucket_list[i].sparse_ind[j] = bucket_list[i].sparse_ind[swapind];
                        bucket_list[i].sparse_ind[swapind] = tmp;
                    }
                    delete[] not_in_list;
                }
            }
        
            //then search in the whole pool, put the redults in the localbuckets
            float pre_tshd = XPC_FCS_CENTER_HWT*params.center_alpha * params.center_alpha;
            //#pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++){
                long begin_index = (thread * num_vec)/num_threads;
                long end_index = ((thread+1) * num_vec)/num_threads;
                float **tmp = (float **)NEW_MAT(params.batch_size, params.center_dim, sizeof(float));
                for (long i = begin_index; i < end_index; i++){
                    float *ptr = vec + vec_size * i;
                    uint64_t sh0 = *((uint64_t *)(&ptr[-16]));
                    uint64_t sh1 = *((uint64_t *)(&ptr[-14]));
                    uint64_t sh2 = *((uint64_t *)(&ptr[-12]));
                    uint64_t sh3 = *((uint64_t *)(&ptr[-10]));
                    float threshold = pre_tshd*ptr[-1];
                    for (long j = 0; j < params.batch_size; j++){
                        for (long k = 0; k < params.center_dim; k++){
                            tmp[j][k] = ptr[bucket_list[j].sparse_ind[k+k]]-ptr[bucket_list[j].sparse_ind[k+k+1]];
                        }
                    }
                    for (long j = 0; j < params.batch_size; j++){
                        if (dot_sse(tmp[j], tmp[j], params.center_dim) > threshold){
                            uint32_t ind = 0;
                            uint32_t *tmp_uint32 = (uint32_t *)tmp[j];
                            for (long k = 0; k < params.center_dim; k++) ind |= ((tmp_uint32[k] & 0x80000000) >> 31) << k;
                            localbucket_list[thread*params.batch_size + j].add(ptr, sh0, sh1, sh2, sh3, ind);
                        }
                    }
                }
                FREE_MAT(tmp);
            }
            
            //put everything in the main buckets
            //#pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++){
                for (long bucket_ind = thread; bucket_ind < params.batch_size; bucket_ind+=num_threads){
                    for (long i = 0; i < (1L << params.center_dim); i++){
                        long count = 0;
                        for (long j = 0; j < num_threads; j++){
                            count += localbucket_list[j*params.batch_size + bucket_ind].num[i];
                        }
                        bucket_list[bucket_ind].num[i] = count;
                        bucket_list[bucket_ind].ptr[i] = (float **) malloc(NPB_VEC_SIZE*count * sizeof(float *));
                        count = 0;
                        for (long j = 0; j < num_threads; j++){
                            for (long k = 0; k < localbucket_list[j*params.batch_size+bucket_ind].num[i]*NPB_VEC_SIZE; k++){
                                bucket_list[bucket_ind].ptr[i][count+k] = localbucket_list[j*params.batch_size+bucket_ind].ptr[i][k];
                            }
                            count += NPB_VEC_SIZE*localbucket_list[j*params.batch_size+bucket_ind].num[i];
                        }
                    }
                }
            }
            for (long i = 0; i < params.batch_size*num_threads; i++)localbucket_list[i].clear();
            gettimeofday(&endb, NULL);
            bucketing_time += (endb.tv_sec-startb.tv_sec)+(double)(endb.tv_usec-startb.tv_usec)/1000000.0;
            long expect_bucket_size = 0;    //used to estimate the number of collisions
            for (long i = 0; i < params.batch_size; i++){
                for (long j = 0; j < (1L<<params.center_dim); j++) expect_bucket_size += bucket_list[i].num[j];
            }
            expect_bucket_size /= params.batch_size;
            cavg_bucket_size += expect_bucket_size;

            /* code for print bucket_list[0], for data analysis
                do {
                    float ***ptr_list = bucket_list[0].ptr;
                    long *sparse_ind = bucket_list[0].sparse_ind;
                    for (long ind = 0; ind < (1L << params.center_dim); ind++){
                        float *tmp = (float *) NEW_VEC(params.center_dim, sizeof(float));
                        std::cout << ind << " " << bucket_list[0].num[ind] << "\n";
                        for (long i = 0; i < bucket_list[0].num[ind]; i++){
                            float *ptr = ptr_list[ind][i];
                            std::cout << *((uint64_t *)(&ptr[-16])) << " " << *((uint64_t *)(&ptr[-14])) << " ";
                            std::cout << *((uint64_t *)(&ptr[-12])) << " " << *((uint64_t *)(&ptr[-10])) << " ";
                            for (long k = 0; k < params.center_dim; k++) tmp[k] = ptr[sparse_ind[k+k]] - ptr[sparse_ind[k+k+1]];
                            print(tmp, params.center_dim);
                            print(ptr, vec_length);
                        }
                        FREE_VEC(tmp);
                    }
                } while (0);
                exit(1);
            */
            
            //search for reductions
            struct timeval starts, ends;
            gettimeofday(&starts, NULL);
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++){
                long num_nbhd = 0;
                uint32_t *nbhd = NULL;
                do {
                    long _22827crt = 1;
                    for (long i = 0; i <= params.search_radius; i++){
                        num_nbhd += _22827crt;
                        _22827crt *= (params.center_dim-i);
                        _22827crt /= i+1;
                    }
                    nbhd = new uint32_t[num_nbhd];
                    long _22827crtind = 0;
                    if (params.search_radius > 4){
                        for (uint32_t a = 0; a < (1L << params.center_dim); a++){
                            if (__builtin_popcountl(a) <= params.search_radius){
                                nbhd[_22827crtind] = a;
                                _22827crtind++;
                            }
                        }
                    }else{
                        if (params.search_radius >= 0){
                            nbhd[_22827crtind] = 0;
                            _22827crtind++;
                        }
                        if (params.search_radius >= 1){
                            for (long i = 0; i < params.center_dim; i++){
                                nbhd[_22827crtind] = (1L << i);
                                _22827crtind++;
                            }
                        }
                        if (params.search_radius >= 2){
                            for (long i = 0; i < params.center_dim; i++){
                                for (long j = i+1; j < params.center_dim; j++){
                                    nbhd[_22827crtind] = (1L << i) + (1L << j);
                                    _22827crtind++;
                                }
                            }
                        }
                        if (params.search_radius >= 3){
                            for (long i = 0; i < params.center_dim; i++){
                                for (long j = i+1; j < params.center_dim; j++){
                                    for (long k = j+1; k < params.center_dim; k++){
                                        nbhd[_22827crtind] = (1L << i) + (1L << j) + (1L << k);
                                        _22827crtind++;
                                    }
                                }
                            }
                        }
                        if (params.search_radius >= 4){
                            for (long i = 0; i < params.center_dim; i++){
                                for (long j = i+1; j < params.center_dim; j++){
                                    for (long k = j+1; k < params.center_dim; k++){
                                        for (long l = k+1; l < params.center_dim; l++){
                                            nbhd[_22827crtind] = (1L << i) + (1L << j) + (1L << k) + (1L << l);
                                            _22827crtind++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } while(0);         //compute nbhds
                long expect_tdl_size = (expect_bucket_size * expect_bucket_size * num_nbhd) / (1L << (params.center_dim+3));
                long expect_tcl_size = (expect_bucket_size * expect_bucket_size * num_nbhd) / (1L << (params.center_dim+8));
                expect_tdl_size -= expect_tdl_size%2;
                expect_tcl_size -= expect_tcl_size%2;
                bool buffer_full = false;
                for (long bucket_ind = thread; bucket_ind < params.batch_size; bucket_ind+=num_threads){
                    if (buffer_full){
                        std::cerr << "[Warning] Pool::fc_Sieve: local_buffer[" << thread << "] full!\n";
                        break;
                    }
                    long num_sh_test = 0;
                    long already_in = 0;
                    long num_tdlp = 0;
                    long num_tclp = 0;
                    long num_tdln = 0;
                    long num_tcln = 0;
                    bool tdlp_full = false;
                    bool tclp_full = false;
                    bool tdln_full = false;
                    bool tcln_full = false;
                    float **to_dot_listp = new float*[expect_tdl_size];
                    float **to_check_listp = new float*[expect_tcl_size];
                    float **to_dot_listn = new float*[expect_tdl_size];
                    float **to_check_listn = new float*[expect_tcl_size];
                    //search for the to_dot_list
                    for (uint32_t ind = 0; ind < (1L<<params.center_dim); ind++){
                        float **iptr_list = bucket_list[bucket_ind].ptr[ind];
                        long inum = bucket_list[bucket_ind].num[ind];
                        //search in the same bucket
                        for (long i = 0; i < inum; i++){
                            if (tdlp_full) break;
                            float *iptr = iptr_list[i*NPB_VEC_SIZE];
                            uint64_t *ish = (uint64_t *)(&iptr_list[i*NPB_VEC_SIZE+1]);
                            for (long j = i+1; j < inum; j++){
                                float *jptr = iptr_list[j*NPB_VEC_SIZE];
                                uint64_t *jsh = (uint64_t *)(&iptr_list[j*NPB_VEC_SIZE+1]);
                                num_sh_test++;
                                long w = __builtin_popcountl(ish[0]^jsh[0]);
                                w += __builtin_popcountl(ish[1]^jsh[1]);
                                w += __builtin_popcountl(ish[2]^jsh[2]);
                                w += __builtin_popcountl(ish[3]^jsh[3]);
                                if (w >= params.XPC_FCS_THRESHOLD) continue;
                                to_dot_listp[num_tdlp] = iptr;
                                to_dot_listp[num_tdlp+1] = jptr;
                                num_tdlp += 2;
                                if (num_tdlp >= expect_tdl_size){
                                    //std::cerr << "[Warning] Pool::fc_Seive: to_dot_list[" << thread << "] full!\n";
                                    tdlp_full = true;
                                    break;
                                }
                            }
                        }
                        //search in the nbhds
                        for (long nbhd_index = 1; nbhd_index < num_nbhd; nbhd_index++){
                            uint32_t indd = ind ^ nbhd[nbhd_index];
                            if (indd < ind) continue;
                            if (tdlp_full) break;
                            float **jptr_list = bucket_list[bucket_ind].ptr[indd];
                            long jnum = bucket_list[bucket_ind].num[indd];
                            for (long i = 0; i < inum; i++){
                                if (tdlp_full) break;
                                float *iptr = iptr_list[i*NPB_VEC_SIZE];
                                uint64_t *ish = (uint64_t *)(&iptr_list[i*NPB_VEC_SIZE+1]);
                                for (long j = 0; j < jnum; j++){
                                    float *jptr = jptr_list[j*NPB_VEC_SIZE];
                                    uint64_t *jsh = (uint64_t *)(&jptr_list[j*NPB_VEC_SIZE+1]);
                                    num_sh_test++;
                                    long w = __builtin_popcountl(ish[0]^jsh[0]);
                                    w += __builtin_popcountl(ish[1]^jsh[1]);
                                    w += __builtin_popcountl(ish[2]^jsh[2]);
                                    w += __builtin_popcountl(ish[3]^jsh[3]);
                                    if (w >= params.XPC_FCS_THRESHOLD) continue;
                                    to_dot_listp[num_tdlp] = iptr;
                                    to_dot_listp[num_tdlp+1] = jptr;
                                    num_tdlp += 2;
                                    if (num_tdlp >= expect_tdl_size){
                                        //std::cerr << "[Warning] Pool::fc_Seive: to_dot_list[" << thread << "] full!\n";
                                        tdlp_full = true;
                                        break;
                                    }
                                }
                            }
                        }
                        //search in the nbhds of diameter points
                        for (long cnbhd_index = 0; cnbhd_index < num_nbhd; cnbhd_index++){
                            uint32_t indd = (uint32_t)((1L<<params.center_dim)-1-ind);
                            indd ^= nbhd[cnbhd_index];
                            if (indd < ind) continue;
                            if (tdln_full) break;
                            float **jptr_list = bucket_list[bucket_ind].ptr[indd];
                            long jnum = bucket_list[bucket_ind].num[indd];
                            for (long i = 0; i < inum; i++){
                                if (tdln_full) break;
                                float *iptr = iptr_list[i*NPB_VEC_SIZE];
                                uint64_t *ish = (uint64_t *)(&iptr_list[i*NPB_VEC_SIZE+1]);
                                for (long j = 0; j < jnum; j++){
                                    float *jptr = jptr_list[j*NPB_VEC_SIZE];
                                    uint64_t *jsh = (uint64_t *)(&jptr_list[j*NPB_VEC_SIZE+1]);
                                    num_sh_test++;
                                    long w = __builtin_popcountl(ish[0]^jsh[0]);
                                    w += __builtin_popcountl(ish[1]^jsh[1]);
                                    w += __builtin_popcountl(ish[2]^jsh[2]);
                                    w += __builtin_popcountl(ish[3]^jsh[3]);
                                    if (w <= 256 - params.XPC_FCS_THRESHOLD) continue;
                                    to_dot_listn[num_tdln] = iptr;
                                    to_dot_listn[num_tdln+1] = jptr;
                                    num_tdln += 2;
                                    if (num_tdln >= expect_tdl_size){
                                        //std::cerr << "[Warning] Pool::fc_Seive: to_dot_list[" << thread << "] full!\n";
                                        tdln_full = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    //search for the to_check_list
                    for (long i = 0; i < num_tdlp; i+=2){
                        float x = dot(to_dot_listp[i], to_dot_listp[i+1], vec_length);
                        float norm = to_dot_listp[i][-1] + to_dot_listp[i+1][-1];
                        if (norm < goal_norm + x + x){
                            to_check_listp[num_tclp] = to_dot_listp[i];
                            to_check_listp[num_tclp+1] = to_dot_listp[i+1];
                            num_tclp += 2;
                            if (num_tclp >= expect_tcl_size){
                                //std::cerr << "[Warning] Pool::fc_Sieve: to_check_list[" << thread << "] full!\n";
                                tclp_full = true;
                                break;
                            }
                        }
                    }
                    for (long i = 0; i < num_tdln; i+=2){
                        float x = dot(to_dot_listn[i], to_dot_listn[i+1], vec_length);
                        float norm = to_dot_listn[i][-1] + to_dot_listn[i+1][-1];
                        if (norm < goal_norm - x - x){
                            to_check_listn[num_tcln] = to_dot_listn[i];
                            to_check_listn[num_tcln+1] = to_dot_listn[i+1];
                            num_tcln += 2;
                            if (num_tcln >= expect_tcl_size){
                                //std::cerr << "[Warning] Pool::fc_Sieve: to_check_list[" << thread << "] full!\n";
                                tcln_full = true;
                                break;
                            }
                        }
                    }

                    //check whether the vectors already in the pool
                    for (long i = 0; i < num_tclp; i+=2){
                        float *iptr = to_check_listp[i];
                        float *jptr = to_check_listp[i+1];
                        uint64_t u = *((uint64_t *)(&iptr[-4]))-*((uint64_t *)(&jptr[-4]));
                        if (uid->check_uid(u)) {
                            already_in++;
                            continue;
                        }
                        if (local_buffer[thread].size == local_buffer[thread].max_size) {
                            buffer_full = true;
                            break;
                        }
                        if (!uid->insert_uid(u)) continue;
                        short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                        sub(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_size);
                        local_buffer[thread].size++;
                    }
                    for (long i = 0; i < num_tcln; i+=2){
                        float *iptr = to_check_listn[i];
                        float *jptr = to_check_listn[i+1];
                        uint64_t u = *((uint64_t *)(&iptr[-4]))+*((uint64_t *)(&jptr[-4]));
                        if (uid->check_uid(u)) {
                            already_in++;
                            continue;
                        }
                        if (local_buffer[thread].size == local_buffer[thread].max_size) {
                            buffer_full = true;
                            break;
                        }
                        if (!uid->insert_uid(u)) continue;
                        short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                        add(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_size);
                        local_buffer[thread].size++;
                    }

                    pthread_spin_lock(&comb_lock);
                    cfound+= (num_tclp + num_tcln)/2;
                    calready_in += already_in;
                    ctest_dot += (num_tdlp + num_tdln)/2;
                    ctest_simhash += num_sh_test;
                    pthread_spin_unlock(&comb_lock);
                    delete[] to_dot_listp;
                    delete[] to_dot_listn;
                    delete[] to_check_listp;
                    delete[] to_check_listn;
                }
                delete[] nbhd;
            }
            gettimeofday(&ends, NULL);
            //pthread_spin_lock(&comb_lock);
            search_time += (ends.tv_sec-starts.tv_sec)+(double)(ends.tv_usec-starts.tv_usec)/1000000.0;
            //pthread_spin_unlock(&comb_lock);

            //check if we have enouth solutions, the same as 3-sieve
            long num_total_sol = 0;
            for (long i = 0; i < num_threads; i++){
                num_total_sol += local_buffer[i].size;
            }
            if ((num_total_sol > params.one_epoch_ratio * num_vec)){
                rel_collection_stop = true;
            }
        }
        if (true){
            printf("num_test_simhash = %ld, num_test_dot = %ld\n", ctest_simhash/(ccount*params.batch_size), ctest_dot/(ccount*params.batch_size));
            printf("[epoch %ld] goal = %ld, num_threads = %ld, bucket_size = %ld, %ld solutions found in %ld buckets, found %ld/%ld, bucketing_time = %fs, search_time = %fs\n", count, (long)sqrt(goal_norm), num_threads, cavg_bucket_size/ccount, cfound-calready_in, ccount*params.batch_size, cfound-calready_in, cfound, bucketing_time, search_time);
        }
        ttt += cfound-calready_in;     

        //put to the main buffer, the same as 3-sieve 
        long num_total_sol = 0;
        for (long i = 0; i < num_threads; i++){
            num_total_sol += local_buffer[i].size;
        }
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++){
            long begin_index = 0;
            for (long j = 0; j < thread; j++){
                begin_index += local_buffer[j].size;
            }
            short *dst = main_buffer.buffer + begin_index * coeff_size;
            short *src = local_buffer[thread].buffer;
            for (long i = 0; i < local_buffer[thread].size; i++){
                copy(dst + i * coeff_size, src + i * coeff_size, coeff_size);
            }
        }
        if (num_total_sol > (num_vec * (1-params.improve_ratio))) num_total_sol = (num_vec * (1-params.improve_ratio));

        //insert to the pool, the same as 3-sieve
        #pragma omp parallel for
        for (long i = 0; i < num_total_sol; i++){
            long *cdst = cvec + cvec_size * (sorted_index-i-1);
            float *dst = (float *)cdst[5];
            short *src = main_buffer.buffer + i * coeff_size;
            if (!uid->erase_uid(*((uint64_t *)(&(dst[-4]))))){
                std::cerr << "something must be wrong with the UidHashTable, warning!\n";
                //std::cout << *(int *)(0x0) << "\n";
            }
            copy((short *)(dst-int_bias), src, coeff_size);
            compute_vec(dst);
            if (dst[-1] > goal_norm*1.00005){
                std::cerr << "warning!";
            }
            if (!uid->safely_check_uid(*((uint64_t *)(&(dst[-4]))))){
                std::cerr << "ssomething must be wrong with the UidHashTable, warning!\n";
                //std::cout << *(int *)(0x0) << "\n";
            }
            
            *((uint64_t *)(&cdst[0])) = *((uint64_t *)(&dst[-16]));
            *((uint64_t *)(&cdst[1])) = *((uint64_t *)(&dst[-14]));
            *((uint64_t *)(&cdst[2])) = *((uint64_t *)(&dst[-12]));
            *((uint64_t *)(&cdst[3])) = *((uint64_t *)(&dst[-10]));
            *((float *)(&cdst[4])) = dst[-1];
        }
                
        sorted_index = sorted_index - num_total_sol;
        if (params.resort_ratio * num_vec > sorted_index){
            sort_cvec();
        }
    }
    return 0;
}

N_ptr_bucket::N_ptr_bucket(long dim, long alloc_size){
    set_params(dim, alloc_size);
}
void N_ptr_bucket::alloc(){
    if (!sparse_ind) sparse_ind = (long *) NEW_VEC(XPC_FCS_CENTER_HWT*dim, sizeof(long));
    if (!ptr) ptr = new float**[1L<<dim];
    if (!num) num = new long[1L<<dim];
    if (!max_size) max_size = new long[1L<<dim];
    for (long i = 0; i < (1L<<dim); i++){
        num[i] = 0;
        max_size[i] = alloc_size;
        ptr[i] = (float **)calloc(NPB_VEC_SIZE*alloc_size*sizeof(float *), 1);
    }
}
void N_ptr_bucket::clear(){
    if (sparse_ind) {
        FREE_VEC((void *)sparse_ind);
        sparse_ind = NULL;
    }
    if (ptr){
        for (long i = 0; i < (1L<<dim); i++){
            free(ptr[i]);
        }
        delete[] ptr;
        ptr = NULL;
    }
    
    if (num) {
        delete[] num;
        num = NULL;
    }
    if (max_size) {
        delete[] max_size;
        max_size = NULL;
    }
}
void N_ptr_bucket::set_params(long dim, long alloc_size){
    this->dim = dim;
    this->alloc_size = alloc_size;
    alloc();
}
void N_ptr_bucket::add(float *vec, uint64_t sh0, uint64_t sh1, uint64_t sh2, uint64_t sh3, long ind){
    if (num[ind] == max_size[ind]) {
        ptr[ind] = (float **)realloc(ptr[ind], NPB_VEC_SIZE*(max_size[ind] + alloc_size)*sizeof(float *));
        max_size[ind] += alloc_size;
    }
    float **dst = &(ptr[ind][num[ind]*NPB_VEC_SIZE]);
    uint64_t *sh = (uint64_t *)(&dst[1]);
    dst[0] = vec;
    sh[0] = sh0;
    sh[1] = sh1;
    sh[2] = sh2;
    sh[3] = sh3;
    num[ind]++;
}
