#include "../include/UidHashTable.h"

UidHashTable::UidHashTable(){
    uid_lock = new padded_spinlock[NUM_UID_LOCK];
    uid_table = new padded_unordered_set[NUM_UID_LOCK];
    n = 0;
    for (long i = 0; i < NUM_UID_LOCK; i++){
		pthread_spin_init(&uid_lock[i].a, PTHREAD_PROCESS_SHARED);
	}
    insert_uid(0);
    uid_coeffs = NULL;
    #if !USE_SPARSEPP
    for (long i = 0; i < NUM_UID_LOCK; i++){
        uid_table[i].a.max_load_factor(8.0);
    }
    #endif
}
UidHashTable::~UidHashTable(){
    if (uid_coeffs != NULL){
        delete[] uid_coeffs;
        uid_coeffs = NULL;
    }
    if (uid_lock) delete[] uid_lock;
    if (uid_table) delete[] uid_table;
    uid_lock = NULL;
    uid_table = NULL;
}
long UidHashTable::size(){
    long sum = 0;
    for (long i = 0; i < NUM_UID_LOCK; i++){
        sum += uid_table[i].a.size();
    }
    return sum;
}

void UidHashTable::reserve_total(long total_hint){
    if (total_hint <= 0) return;
    long per_table = (total_hint + NUM_UID_LOCK - 1) / NUM_UID_LOCK;
    if (per_table < 1) per_table = 1;
    #pragma omp parallel for
    for (long i = 0; i < NUM_UID_LOCK; i++){
        uid_table[i].a.reserve(per_table);
    }
}

#if !UID_OP_INLINE
bool UidHashTable::insert_uid(UidType uid){
    normalize_uid(uid);
    pthread_spin_lock(&uid_lock[uid % NUM_UID_LOCK].a);
    bool success = uid_table[uid % NUM_UID_LOCK].a.insert(uid).second;
    pthread_spin_unlock(&uid_lock[uid % NUM_UID_LOCK].a);
    return success;
}
bool UidHashTable::erase_uid(UidType uid){
    if (uid == 0) return false;
    normalize_uid(uid);
    pthread_spin_lock(&uid_lock[uid % NUM_UID_LOCK].a);
    bool success = (uid_table[uid % NUM_UID_LOCK].a.erase(uid) != 0);
    pthread_spin_unlock(&uid_lock[uid % NUM_UID_LOCK].a);
    return success;
}
bool UidHashTable::check_uid(UidType uid){
    normalize_uid(uid);
    if (uid_table[uid % NUM_UID_LOCK].a.count(uid) != 0) return true;
    return false;
}
bool UidHashTable::safely_check_uid(UidType uid){
    normalize_uid(uid);
    pthread_spin_lock(&uid_lock[uid % NUM_UID_LOCK].a);
    bool success = (uid_table[uid % NUM_UID_LOCK].a.count(uid) != 0);
    pthread_spin_unlock(&uid_lock[uid % NUM_UID_LOCK].a);
    return success;
}
#endif
