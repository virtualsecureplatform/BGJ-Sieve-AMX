/** 
 * \warning: old codes based on float32, do not read it
*/
#include "../include/pool.h"

#define XPC_TS_THRESHOLD 96
#define XPC_TS_TRD_THRESHOLD 140
#define XPC_TS_BUCKET_THRESHOLD 102

int Pool::three_Sieve(three_Sieve_params params, int show_details, long maxcc){
    bool THREE_SIEVE_STUCK = false;
    long coeff_size = (int_bias-16)*2;
    long coeff_block = coeff_size/32;
#ifndef __ON_CYGWIN
    cpu_set_t *mask = new cpu_set_t[num_threads];
    for (long i = 0; i < num_threads; i++){
        CPU_ZERO(&mask[i]);
        CPU_SET(i, &mask[i]);
    }
#endif
    if (show_details == 1){
        std::cerr << "\n";
        std::cerr << "begin 3Sieve on context ["<< index_l<<", "<<index_r << "], gh = "<< sqrt(gh2);
        std::cerr << ", pool size = "<< num_vec<<", "<<num_threads<<" threads will be used";
    }
    sort_cvec();

    long count = -1;
    while (!sieve_is_over(params.saturation_radius, params.saturation_ratio, (show_details == 2))){
        if (THREE_SIEVE_STUCK) break;
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
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++){
#ifndef __ON_CYGWIN
            pthread_setaffinity_np(pthread_self(), sizeof(mask[omp_get_thread_num()]), &mask[omp_get_thread_num()]);
#endif
            long ccount = 0;
            //long avg_bucket_size = 0;
            //long already_in = 0;
            //long found = 0;
            //long already_in3 = 0;
            //long found3 = 0;
            //long pass = 0;
            //long passb = 0;
            //long pass3 = 0;
            __attribute__ ((aligned (64))) float tmp_store[vec_length];
            float *tmp = &tmp_store[0];
            float alpha2 = params.alpha * 2.0;
            long last_num_total_sol = 0;
            long stucktime = 0;
            while(!rel_collection_stop){

                ccount++;
                if (ccount > maxcc) {
                    THREE_SIEVE_STUCK = true;
                    break;
                }
                //centering
                long index = Uniform_long(num_vec);
                long *cptr = cvec+index*cvec_size;
                float *ptr = (float *) cptr[5];
                long **bucketp = new long*[num_vec];
                long **bucketn = new long*[num_vec];
                float *dotp = new float[num_vec];           //we store 2 * dot_product here!
                float *dotn = new float[num_vec];           //we store 2 * dot_product here!
                long num_p = 1;
                long num_n = 0;


                //bucketing
                long *cpptr = cvec;
                set_zero(tmp, vec_length);
                red(tmp, ptr, -1.0/params.alpha, vec_length);
                long max_bucket_size = sqrt(num_vec)*5.0;
                for (long i = 0; i < num_vec; i++){
                    long w = __builtin_popcountl((*((uint64_t *)(&cptr[0]))) ^ (*((uint64_t *)(&cpptr[0]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[1]))) ^ (*((uint64_t *)(&cpptr[1]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[2]))) ^ (*((uint64_t *)(&cpptr[2]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[3]))) ^ (*((uint64_t *)(&cpptr[3]))));
                    if (w < XPC_TS_BUCKET_THRESHOLD || w > (256 - XPC_TS_BUCKET_THRESHOLD)){
                        //passb++;
                        float *pptr = (float *)cpptr[5];
                        float x = dot(pptr, tmp, vec_length);
                        if (fabs(x)>pptr[-1]){
                            if (fabs(x) > pptr[-1]*3.0){
                                if (i == index){
                                    bucketp[0] = cptr;
                                    dotp[0] = x * alpha2;
                                }
                            }else{
                                if (x > 0){
                                    bucketp[num_p] = cpptr;
                                    dotp[num_p] = x * alpha2;
                                    num_p++;
                                }else{
                                    bucketn[num_n] = cpptr;
                                    dotn[num_n] = x * alpha2;
                                    num_n++;
                                }
                                if (num_n + num_p > max_bucket_size) break;
                            }
                        }
                    }
                    cpptr += cvec_size;
                }
                //avg_bucket_size += num_p + num_n;
                //long old_num_sol = local_buffer[thread].size;


                //search the reductions
                //two reductions wrt the center
                do {
                    float ib = ptr[-1] - goal_norm;
                    for (long i = 1; i < num_p; i++){
                        long *icptr = bucketp[i];
                        if (dotp[i]>(ib + *((float *)(&icptr[4])))){
                            //found++;
                            float *iptr = (float *)icptr[5];
                            uint64_t u = (*((uint64_t *)(&ptr[-4]))-*((uint64_t *)(&iptr[-4])));
                            /*if (uid->check_uid(u)) {
                                //already_in++;
                                continue;
                            }*/
                            if (local_buffer[thread].size == local_buffer[thread].max_size) break;
                            if (!uid->insert_uid(u)) continue;
                            short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                            sub(dst, (short*)(ptr-int_bias), (short *)(iptr-int_bias), coeff_size);
                            local_buffer[thread].size++;
                        }
                    }
                    for (long i = 0; i < num_n; i++){
                        long *icptr = bucketn[i];
                        if ( (-dotn[i]) > (ib + *((float *)(&icptr[4]))) ){
                            //found++;
                            float *iptr = (float *)icptr[5];
                            uint64_t u = (*((uint64_t *)(&ptr[-4]))+*((uint64_t *)(&iptr[-4])));
                            /*if (uid->check_uid(u)) {
                                //already_in++;
                                continue;
                            }*/
                            if (local_buffer[thread].size == local_buffer[thread].max_size) break;
                            if (!uid->insert_uid(u)) continue;
                            short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                            add(dst, (short*)(ptr-int_bias), (short *)(iptr-int_bias), coeff_size);
                            local_buffer[thread].size++;
                        }
                    }
                } while(0);
                //pp-reduction
                for (long i = 1; i < num_p; i++){
                    if (local_buffer[thread].size>local_buffer[thread].max_size*0.9) break;
                    long *icptr = bucketp[i];
                    float *iptr = (float *)icptr[5];
                    const float ib3 = ptr[-1] + iptr[-1] - goal_norm - dotp[i];
                    const float ib2 = iptr[-1] - goal_norm;
                    copy(tmp, iptr, vec_length);
                    add(tmp, iptr, vec_length);
                    for (long j = i+1; j < num_p; j++){
                        long *jcptr = bucketp[j];
                        long w = __builtin_popcountl((*((uint64_t *)(&icptr[0]))) ^ (*((uint64_t *)(&jcptr[0]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[1]))) ^ (*((uint64_t *)(&jcptr[1]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[2]))) ^ (*((uint64_t *)(&jcptr[2]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[3]))) ^ (*((uint64_t *)(&jcptr[3]))));
                        //triple_reductoin
                        if (w > XPC_TS_TRD_THRESHOLD){
                            //pass3++;
                            float *jptr = (float *)(jcptr[5]);
                            float x = dot(tmp, jptr, vec_length);
                            if ((ib3+*((float *)(&jcptr[4]))) < (dotp[j] - x)){
                                //found3++;
                                uint64_t u = (*((uint64_t *)(&ptr[-4]))-*((uint64_t *)(&iptr[-4]))-*((uint64_t *)(&jptr[-4])));
                                /*if (uid->check_uid(u)) {
                                    //already_in3++;
                                    continue;
                                }*/
                                if (local_buffer[thread].size == local_buffer[thread].max_size) break;
                                if (!uid->insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                sub(dst, (short*)(ptr-int_bias), (short *)(iptr-int_bias), coeff_size);
                                sub(dst, (short*)(jptr-int_bias), coeff_size);
                                local_buffer[thread].size++;
                            }
                        }
                        //two_reduction
                        if (w < XPC_TS_THRESHOLD){
                            //pass++;
                            float *jptr = (float *)(jcptr[5]);
                            float x = dot(tmp, jptr, vec_length);
                            if ((ib2 + *((float *)(&jcptr[4])))<x){
                                //found++;
                                uint64_t u = *((uint64_t *)(&iptr[-4]))-*((uint64_t *)(&jptr[-4]));
                                /*if (uid->check_uid(u)) {
                                    //already_in++;
                                    continue;
                                }*/
                                if (local_buffer[thread].size == local_buffer[thread].max_size) break;
                                if (!uid->insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                sub(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_size);
                                local_buffer[thread].size++;
                            }
                        }
                    }
                }
                //nn-reduction
                for (long i = 0; i < num_n; i++){
                    if (local_buffer[thread].size>local_buffer[thread].max_size*0.9) break;
                    long *icptr = bucketn[i];
                    float *iptr = (float *)icptr[5];
                    const float ib3 = ptr[-1] + iptr[-1] - goal_norm + dotn[i];
                    const float ib2 = iptr[-1] - goal_norm;
                    copy(tmp, iptr, vec_length);
                    add(tmp, iptr, vec_length);
                    for (long j = i+1; j < num_n; j++){
                        long *jcptr = bucketn[j];
                        long w = __builtin_popcountl((*((uint64_t *)(&icptr[0]))) ^ (*((uint64_t *)(&jcptr[0]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[1]))) ^ (*((uint64_t *)(&jcptr[1]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[2]))) ^ (*((uint64_t *)(&jcptr[2]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[3]))) ^ (*((uint64_t *)(&jcptr[3]))));
                        //triple_reductoin
                        if (w > XPC_TS_TRD_THRESHOLD){
                            //pass3++;
                            float *jptr = (float *)(jcptr[5]);
                            float x = dot(tmp, jptr, vec_length);
                            if ((ib3+*((float *)(&jcptr[4]))) < (-dotn[j] - x)){
                                //found3++;
                                uint64_t u = (*((uint64_t *)(&ptr[-4]))+*((uint64_t *)(&iptr[-4]))+*((uint64_t *)(&jptr[-4])));
                                /*if (uid->check_uid(u)) {
                                    //already_in3++;
                                    continue;
                                }*/
                                if (local_buffer[thread].size == local_buffer[thread].max_size) break;
                                if (!uid->insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                add(dst, (short*)(ptr-int_bias), (short *)(iptr-int_bias), coeff_size);
                                add(dst, (short*)(jptr-int_bias), coeff_size);
                                local_buffer[thread].size++;
                            }
                        }
                        //two_reduction
                        if (w < XPC_TS_THRESHOLD){
                            //pass++;
                            float *jptr = (float *)(jcptr[5]);
                            float x = dot(tmp, jptr, vec_length);
                            if ((ib2 + *((float *)(&jcptr[4])))<x){
                                //found++;
                                uint64_t u = *((uint64_t *)(&iptr[-4]))-*((uint64_t *)(&jptr[-4]));
                                /*if (uid->check_uid(u)) {
                                    //already_in++;
                                    continue;
                                }*/
                                if (local_buffer[thread].size == local_buffer[thread].max_size) break;
                                if (!uid->insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                sub(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_size);
                                local_buffer[thread].size++;
                            }
                        }
                    }
                }
                //np-reduction
                for (long i = 1; i < num_p; i++){
                    if (local_buffer[thread].size>local_buffer[thread].max_size*0.9) break;
                    long *icptr = bucketp[i];
                    float *iptr = (float *)icptr[5];
                    const float ib3 = ptr[-1] + iptr[-1] - goal_norm - dotp[i];
                    const float ib2 = iptr[-1] - goal_norm;
                    copy(tmp, iptr, vec_length);
                    add(tmp, iptr, vec_length);
                    for (long j = 0; j < num_n; j++){
                        long *jcptr = bucketn[j];
                        long w = __builtin_popcountl((*((uint64_t *)(&icptr[0]))) ^ (*((uint64_t *)(&jcptr[0]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[1]))) ^ (*((uint64_t *)(&jcptr[1]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[2]))) ^ (*((uint64_t *)(&jcptr[2]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[3]))) ^ (*((uint64_t *)(&jcptr[3]))));
                        //triple_reduction
                        if (w < (256 - XPC_TS_TRD_THRESHOLD)){
                            //pass3++;
                            float *jptr = (float *)(jcptr[5]);
                            float x = dot(tmp, jptr, vec_length);
                            if ((ib3+*((float *)(&jcptr[4]))) < (-dotn[j] + x)){
                                //found3++;
                                uint64_t u = (*((uint64_t *)(&ptr[-4]))-*((uint64_t *)(&iptr[-4]))+*((uint64_t *)(&jptr[-4])));
                                /*if (uid->check_uid(u)) {
                                    //already_in3++;
                                    continue;
                                }*/
                                if (local_buffer[thread].size == local_buffer[thread].max_size) break;
                                if (!uid->insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                sub(dst, (short*)(ptr-int_bias), (short *)(iptr-int_bias), coeff_size);
                                add(dst, (short*)(jptr-int_bias), coeff_size);
                                local_buffer[thread].size++;
                            }
                        }
                        //two_reduction
                        if (w > (256 - XPC_TS_THRESHOLD)){
                            //pass++;
                            float *jptr = (float *)(jcptr[5]);
                            float x = dot(tmp, jptr, vec_length);
                            if ((ib2 + *((float *)(&jcptr[4])))<-x){
                                //found++;
                                uint64_t u = *((uint64_t *)(&iptr[-4]))+*((uint64_t *)(&jptr[-4]));
                                /*if (uid->check_uid(u)) {
                                    //already_in++;
                                    continue;
                                }*/
                                if (local_buffer[thread].size == local_buffer[thread].max_size) break;
                                if (!uid->insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                add(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_size);
                                local_buffer[thread].size++;
                            }
                        }
                    }
                }
                delete[] bucketn;
                delete[] bucketp;
                delete[] dotn;
                delete[] dotp;

                long num_total_sol = 0;
                for (long i = 0; i < num_threads; i++){
                    num_total_sol += local_buffer[i].size;
                }
                if (num_total_sol < last_num_total_sol+ stucktime/3){
                    stucktime++;
                    if (stucktime > 64) {
                        THREE_SIEVE_STUCK = true;
                        break;
                    }
                }else{
                    stucktime = 0;
                    last_num_total_sol = num_total_sol;
                }
                if ((num_total_sol > params.one_epoch_ratio * num_vec)){
                    rel_collection_stop = true;
                }            
            }
            if (CSD > 40 && !THREE_SIEVE_STUCK){
                if (show_details == 1){if (thread == 0) std::cerr << ".";}
                //printf("[epoch %ld] goal = %ld, bucket_size = %ld, %ld solutions found in %ld buckets, pass = %ld, pass3 = %ld, already in ratio = %f, already in3 ratio = %f, found = %ld, found3 = %ld\n", count, (long)sqrt(goal_norm), avg_bucket_size/ccount, local_buffer[thread].size,ccount, pass/ccount, pass3/ccount, (already_in+.0)/found, (already_in3+.0)/found3, found, found3);
            }
        }

        //put to the main buffer
        long num_total_sol = 0;
        for (long i = 0; i < num_threads; i++){
            num_total_sol += local_buffer[i].size;
        }
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++){
#ifndef __ON_CYGWIN
            pthread_setaffinity_np(pthread_self(), sizeof(mask[omp_get_thread_num()]), &mask[omp_get_thread_num()]);
#endif
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

        

        //insert to the pool
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
#ifndef __ON_CYGWIN
    delete[] mask;
#endif
    if (THREE_SIEVE_STUCK) {
        std::cerr << "[Warning] Pool::three_Sieve: get stucked, sieving aborted!";
        return THREE_SIEVE_GET_STUCKED;
    }
    return 0;
    //if (!check_pool_status()){std::cerr << *(int *)(0x0) << "\n";}
}

int Pool::left_progressive_3sieve(long ind_l, long ind_r, long num_threads, int show_details){
    if (ind_r - ind_l < 40) {
        std::cerr << "[Error] Pool::left_progressive_sieve: sieving dim too small, aborted.\n";
        return 0;
    }
    clear_pool();
    set_num_threads(num_threads);
    set_MSD(ind_r - ind_l);
    set_max_pool_size((long)(pow(4./3., MSD*0.5) * 3.2) + 1);
    set_sieving_context(max(ind_r-40, ind_l), ind_r);
    gaussian_sampling((long)(pow(4./3., CSD * 0.5) * 3.2));
    three_Sieve_params p;
    p.alpha = 0.315;

    three_Sieve(p, show_details, 15000);
    while(index_l > ind_l){
        extend_left();
        gaussian_sampling((long)(pow(4./3., CSD * 0.5) * 3.2));
        if (three_Sieve(p, show_details, 15000) == THREE_SIEVE_GET_STUCKED) return 0;
    }
    if (show_details == 2){
        printf("\r");
    }
    if (show_details == 1){
        std::cout << std::endl;
    }
    return 1;
}

int Pool::left_progressive_bgj1sieve(long ind_l, long ind_r, long num_threads, long log_level){
    if (ind_r - ind_l < 40) {
        std::cerr << "[Error] Pool::left_progressive_sieve: sieving dim too small, aborted.\n";
        return 0;
    }
    clear_pool();
    set_num_threads(num_threads);
    set_MSD(ind_r - ind_l);
    set_max_pool_size((long)(pow(4./3., MSD*0.5) * 3.2) + 1);
    set_sieving_context(max(ind_r-40, ind_l), ind_r);
    gaussian_sampling((long)(pow(4./3., CSD * 0.5) * 3.2));

    bgj1_Sieve(log_level, 1);
    while(index_l > ind_l){
        extend_left();
        //gaussian_sampling((long)(pow(4./3., CSD * 0.5) * 3.2));
        if (bgj1_Sieve(log_level, 1, (long)(pow(4./3., CSD * 0.5) * 3.2) - num_vec)) return 0;
    }
    return 1;
}
int Pool::left_progressive_bgj2sieve(long ind_l, long ind_r, long num_threads, long log_level){
    if (ind_r - ind_l < 40) {
        std::cerr << "[Error] Pool::left_progressive_sieve: sieving dim too small, aborted.\n";
        return 0;
    }
    clear_pool();
    set_num_threads(num_threads);
    set_MSD(ind_r - ind_l);
    set_max_pool_size((long)(pow(4./3., MSD*0.5) * 3.2) + 1);
    set_sieving_context(max(ind_r-40, ind_l), ind_r);
    gaussian_sampling((long)(pow(4./3., CSD * 0.5) * 3.2));

    bgj2_Sieve(log_level, 1);
    while(index_l > ind_l){
        extend_left();
        //gaussian_sampling((long)(pow(4./3., CSD * 0.5) * 3.2));
        if (bgj2_Sieve(log_level, 1, (long)(pow(4./3., CSD * 0.5) * 3.2) - num_vec)) return 0;
    }
    return 1;
}
