#ifndef __UIDHASHTABLE_H
#define __UIDHASHTABLE_H

#include <pthread.h>
#include <array>
#include <type_traits>
#include <limits>
#include "config.h"
#include "utils.h"
#include "sampler.h"

/***\
*   I learned this implementation of UidHashTable from G6K, the code is similar
*
*   Copyright (C) 2018-2021 Team G6K
*
*   This file is part of G6K. G6K is free software:
*   you can redistribute it and/or modify it under the terms of the
*   GNU General Public License as published by the Free Software Foundation,
*   either version 2 of the License, or (at your option) any later version.
*
*   G6K is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with G6K. If not, see <http://www.gnu.org/licenses/>.
*
****/


using UidType = uint64_t;
#if USE_GTL_UID_TABLE
#include "../dep/gtl/include/gtl/phmap.hpp"
struct padded_unordered_set { __attribute__ ((aligned(128))) gtl::flat_hash_set<UidType> a; };
#elif USE_SPARSEPP
#include "../dep/sparsepp/spp.h"
struct padded_unordered_set { __attribute__ ((aligned(128))) spp::sparse_hash_set<UidType> a; };
#else
#include <unordered_set>
struct padded_unordered_set { __attribute__ ((aligned(128))) std::unordered_set<UidType> a; };
#endif
struct padded_spinlock { __attribute__ ((aligned(128))) pthread_spinlock_t a; };


class UidHashTable {
    public:
        uint64_t *uid_coeffs = NULL;
        UidHashTable();
        ~UidHashTable();
        #if !UID_OP_INLINE
        bool insert_uid(UidType uid);
        bool erase_uid(UidType uid);
        bool check_uid(UidType uid);
        bool safely_check_uid(UidType uid);
        #else
        inline bool insert_uid(UidType uid);
        inline bool erase_uid(UidType uid);
        inline bool check_uid(UidType uid);
        inline bool safely_check_uid(UidType uid);
        #endif
        void reserve_total(long total_hint);
        inline void reset_hash_function(long CSD);
        long size();

    // private:
        inline void normalize_uid(UidType &uid);
        static constexpr unsigned NUM_UID_LOCK = 8192;
        padded_spinlock *uid_lock = NULL;
        padded_unordered_set *uid_table = NULL;
        long n;                         //current dimension   
};

inline void UidHashTable::normalize_uid(UidType &uid){
    if (uid > std::numeric_limits<UidType>::max()/2  + 1) uid = -uid;
}
inline void UidHashTable::reset_hash_function(long CSD){
    delete[] uid_coeffs;
    uid_coeffs = new uint64_t[CSD+16];
    for (long i = 0; i < CSD; i++){
        uid_coeffs[i] = Uniform_u64();
    }
    #pragma omp parallel for
    for (long i = 0; i < NUM_UID_LOCK; i++){
        uid_table[i].a.clear();
    }
    insert_uid(0);
}

#if UID_OP_INLINE
inline bool UidHashTable::insert_uid(UidType uid){
    normalize_uid(uid);
    pthread_spin_lock(&uid_lock[uid % NUM_UID_LOCK].a);
    bool success = uid_table[uid % NUM_UID_LOCK].a.insert(uid).second;
    pthread_spin_unlock(&uid_lock[uid % NUM_UID_LOCK].a);
    return success;
}
inline bool UidHashTable::erase_uid(UidType uid){
    if (uid == 0) return false;
    normalize_uid(uid);
    pthread_spin_lock(&uid_lock[uid % NUM_UID_LOCK].a);
    bool success = (uid_table[uid % NUM_UID_LOCK].a.erase(uid) != 0);
    pthread_spin_unlock(&uid_lock[uid % NUM_UID_LOCK].a);
    return success;
}
inline bool UidHashTable::check_uid(UidType uid){
    normalize_uid(uid);
    if (uid_table[uid % NUM_UID_LOCK].a.count(uid) != 0) return true;
    return false;
}
inline bool UidHashTable::safely_check_uid(UidType uid){
    normalize_uid(uid);
    pthread_spin_lock(&uid_lock[uid % NUM_UID_LOCK].a);
    bool success = (uid_table[uid % NUM_UID_LOCK].a.count(uid) != 0);
    pthread_spin_unlock(&uid_lock[uid % NUM_UID_LOCK].a);
    return success;
}
#endif


#endif
