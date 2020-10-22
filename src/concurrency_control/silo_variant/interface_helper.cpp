/**
 * @file helper.cpp
 */

#include "concurrency_control/silo_variant//include/interface_helper.h"

#include "concurrency_control/silo_variant/include/garbage_collection.h"
#include "concurrency_control/silo_variant/include/session_info_table.h"

#ifdef INDEX_KOHLER_MASSTREE
#include "index/masstree_beta/include/masstree_beta_wrapper.h"
#endif

#include "boost/filesystem.hpp"
#include "include/tuple_local.h"

namespace shirakami::cc_silo_variant {

Status enter(Token &token) {  // NOLINT
    Status ret_status = session_info_table::decide_token(token);
#ifdef INDEX_YAKUSHIMA
    yakushima::Token kvs_token{};
    yakushima::enter(kvs_token);
    static_cast<session_info*>(token)->set_kvs_token(kvs_token);
#endif
    return ret_status;
}

void fin() {
    delete_all_records();
    garbage_collection::release_all_heap_objects();

    // Stop DB operation.
    epoch::set_epoch_thread_end(true);
    epoch::join_epoch_thread();
    session_info_table::fin_kThreadTable();

#ifdef INDEX_YAKUSHIMA
    yakushima::fin();
#endif
}

Status init(const std::string_view log_directory_path) {  // NOLINT
    /**
     * The default value of log_directory is PROJECT_ROOT.
     */
    Log::set_kLogDirectory(log_directory_path);
    if (log_directory_path == MAC2STR(PROJECT_ROOT)) {
        Log::get_kLogDirectory().append("/log");
    }

    /**
     * check whether log_directory_path is filesystem objects.
     */
    boost::filesystem::path log_dir{Log::get_kLogDirectory()};
    if (boost::filesystem::exists(log_dir)) {
        /**
         * some file exists.
         * check whether it is directory.
         */
        if (!boost::filesystem::is_directory(log_dir)) {
            return Status::ERR_INVALID_ARGS;
        }
    } else {
        /**
         * directory which has log_directory_path as a file path doesn't exist.
         * it can create.
         */
        boost::filesystem::create_directories(log_dir);
    }

    /**
     * If it already exists log files, it recoveries from those.
     */
    // single_recovery_from_log();
    session_info_table::init_kThreadTable();
    epoch::invoke_epocher();
#ifdef INDEX_YAKUSHIMA
    yakushima::init();
#endif

    return Status::OK;
}

Status leave(Token const token) {  // NOLINT
    for (auto &&itr : session_info_table::get_thread_info_table()) {
        if (&itr == static_cast<session_info*>(token)) {
            if (itr.get_visible()) {
#ifdef INDEX_YAKUSHIMA
                yakushima::leave(
                        static_cast<session_info*>(token)->get_yakushima_token());
#endif
                itr.set_visible(false);
                return Status::OK;
            }
            return Status::WARN_NOT_IN_A_SESSION;
        }
    }
    return Status::ERR_INVALID_ARGS;
}

void tx_begin(Token const token) {
    auto* ti = static_cast<session_info*>(token);
    ti->set_tx_began(true);
    ti->set_epoch(epoch::load_acquire_global_epoch());
}

Status read_record(Record &res, const Record* const dest) {  // NOLINT
    tid_word f_check;
    tid_word s_check;  // first_check, second_check for occ

    f_check.set_obj(loadAcquire(dest->get_tidw().get_obj()));

    for (;;) {
        while (f_check.get_lock()) {
            f_check.set_obj(loadAcquire(dest->get_tidw().get_obj()));
        }

        if (f_check.get_absent()) {
            return Status::WARN_CONCURRENT_DELETE;
            // other thread is inserting this record concurrently,
            // but it isn't committed yet.
        }

        res.get_tuple() = dest->get_tuple();  // execute copy assign.

        s_check.set_obj(loadAcquire(dest->get_tidw().get_obj()));
        if (f_check == s_check) {
            break;
        }
        f_check = s_check;
    }

    res.set_tidw(f_check);
    return Status::OK;
}

void write_phase(session_info* const ti, const tid_word &max_r_set,
                 const tid_word &max_w_set) {
#ifdef INDEX_KOHLER_MASSTREE
    masstree_wrapper<Record>::thread_init(sched_getcpu());
#endif  // INDEX_KOHLER_MASSTREE

    /*
     * It calculates the smallest number that is
     * (a) larger than the TID of any record read or written by the transaction,
     * (b) larger than the worker's most recently chosen TID,
     * and (C) in the current global epoch.
     */
    tid_word tid_a;
    tid_word tid_b;
    tid_word tid_c;

    /*
     * calculates (a)
     * about read_set
     */
    tid_a = std::max(max_w_set, max_r_set);
    tid_a.inc_tid();

    /*
     * calculates (b)
     * larger than the worker's most recently chosen TID,
     */
    tid_b = ti->get_mrctid();
    tid_b.inc_tid();

    /* calculates (c) */
    tid_c.set_epoch(ti->get_epoch());

    /* compare a, b, c */
    tid_word max_tid = std::max({tid_a, tid_b, tid_c});
    max_tid.set_lock(false);
    max_tid.set_absent(false);
    max_tid.set_latest(true);
    max_tid.set_epoch(ti->get_epoch());
    ti->set_mrc_tid(max_tid);

#ifdef WAL
    ti->wal(max_tid.get_obj());
#endif

    for (auto iws = ti->get_write_set().begin(); iws != ti->get_write_set().end();
         ++iws) {
        Record* rec_ptr = iws->get_rec_ptr();
        switch (iws->get_op()) {
            case OP_TYPE::UPDATE: {
                std::string* old_value{};
                std::string_view new_value_view =
                        iws->get_tuple(iws->get_op()).get_value();
                rec_ptr->get_tuple().get_pimpl()->set_value(
                        new_value_view.data(), new_value_view.size(), &old_value);
                storeRelease(rec_ptr->get_tidw().get_obj(), max_tid.get_obj());
                if (old_value != nullptr) {
                    ti->get_gc_value_container()->emplace_back(
                            std::make_pair(old_value, ti->get_epoch()));
                }
                break;
            }
            case OP_TYPE::INSERT: {
                storeRelease(rec_ptr->get_tidw().get_obj(), max_tid.get_obj());
                break;
            }
            case OP_TYPE::DELETE: {
                tid_word delete_tid = max_tid;
                delete_tid.set_absent(true);
                std::string_view key_view = rec_ptr->get_tuple().get_key();
#ifdef INDEX_KOHLER_MASSTREE
                kohler_masstree::get_mtdb().remove_value(key_view.data(),
                                                         key_view.size());
#elif INDEX_YAKUSHIMA
                yakushima::remove(ti->get_yakushima_token(), key_view);
#endif
                storeRelease(rec_ptr->get_tidw().get_obj(), delete_tid.get_obj());

                /**
                 * create information for garbage collection.
                 */
                ti->get_gc_record_container()->emplace_back(rec_ptr);

                break;
            }
            default:
                std::cout << __FILE__ << " : " << __LINE__ << " : fatal error."
                          << std::endl;
                std::abort();
        }
    }

    /**
     * about holding operation info.
     */
    ti->clean_up_ops_set();
    /**
     * about scan operation.
     */
    ti->clean_up_scan_caches();

    ti->gc_records_and_values();
}
}  // namespace shirakami::cc_silo_variant
