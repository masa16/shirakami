/**
 * @file garbage_collection.h
 * @brief about garbage collection
 */

#pragma once

#include <array>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "cpu.h"
#include "epoch.h"
#include "record.h"

namespace shirakami::cc_silo_variant::garbage_collection {

alignas(CACHE_LINE_SIZE) inline std::array<  // NOLINT
        std::vector<Record*>,
        KVS_NUMBER_OF_LOGICAL_CORES> kGarbageRecords;  // NOLINT
alignas(CACHE_LINE_SIZE) inline std::array<                         // NOLINT
        std::vector<std::pair<std::string*, epoch::epoch_t>>,
        KVS_NUMBER_OF_LOGICAL_CORES> kGarbageValues;                   // NOLINT

/**
 * @brief Delete std::vector<Record*> kGarbageRecords at
 * shirakami/src/gcollection.concurrency_control
 * @pre This function should be called at terminating db.
 * @return void
 */
extern void delete_all_garbage_records();

/**
 * @brief Delete first of std::pair<std::string*, epoch_t>> kGarbageValues at
 * shirakami/src/gcollection.concurrency_control
 * @pre This function should be called at terminating db.
 * @return void
 */
extern void delete_all_garbage_values();

[[maybe_unused]] static std::vector<Record*> &get_garbage_records_at(  // NOLINT
        std::size_t index) {
    return kGarbageRecords.at(index);
}

[[maybe_unused]] static std::vector<std::pair<std::string*, epoch::epoch_t>> &
get_garbage_values_at(std::size_t index) {  // NOLINT
    return kGarbageValues.at(index);
}

/**
 * @brief Release all heap objects in this system.
 * @details Do three functions: delete_all_garbage_values(),
 * delete_all_garbage_records(), and remove_all_leaf_from_mt_db_and_release().
 * @pre This function should be called at terminating db.
 * @return void
 */
[[maybe_unused]] extern void release_all_heap_objects();

/**
 * @brief Remove all leaf nodes from MTDB and release those heap objects.
 * @pre This function should be called at terminating db.
 * @return void
 */
extern void remove_all_leaf_from_mt_db_and_release();

}  // namespace shirakami::cc_silo_variant::garbage_collection
