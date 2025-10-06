/*
 * Copyright 2019-2024 tsurugi project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <xmmintrin.h>

#include <cstring>
#include <cstdint>

// shirakami/test
#include "result_ltx.h"

// shirakami/bench
#include "build_db.h"
#include "gen_key.h"
#include "ycsb_ltx/include/gen_tx.h"

// shirakami/src/include
#include "atomic_wrapper.h"

#include "clock.h"
#include "compiler.h"
#include "cpu.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "concurrency_control/include/session.h"

#include "shirakami/interface.h"

#include "boost/filesystem.hpp"


using namespace shirakami;

/**
 * general option.
 */
DEFINE_uint64( // NOLINT
        cpumhz, 2100,
        "# cpu MHz of execution environment. It is used measuring some "
        "time.");
DEFINE_uint64(duration, 1, "Duration of benchmark in seconds.");       // NOLINT
DEFINE_uint64(key_length, 8, "# length of value(payload). min is 8."); // NOLINT
DEFINE_uint64(ops, 1, "# operations per a transaction.");              // NOLINT
DEFINE_string(ops_read_type, "point", "type of read operation.");      // NOLINT
DEFINE_string(ops_write_type, "update", "type of write operation.");   // NOLINT
DEFINE_uint64(record, 10, "# database records(tuples).");              // NOLINT
DEFINE_uint64(rratio, 100, "rate of reads in a transaction.");         // NOLINT
DEFINE_uint64(scan_length, 100, "number of records in scan range.");   // NOLINT
DEFINE_double(skew, 0.0, "access skew of transaction.");               // NOLINT
DEFINE_double(ltx_ratio, 0.0, "ratio of LTX.");                        // NOLINT
DEFINE_uint64(ltx_ops, 1, "operetions per tx for LTX.");               // NOLINT
DEFINE_string(ltx_read_type, "off", "type of read operation in LTX."); // NOLINT
DEFINE_uint64(ltx_rratio, 100, "rate of reads in a LTX.");             // NOLINT
DEFINE_bool(sparse_key, false, "sparse key.");                            // NOLINT
DEFINE_uint64(thread, 1, "# worker threads.");                         // NOLINT
DEFINE_string(transaction_type, "short", "type of transaction.");      // NOLINT
DEFINE_uint64(val_length, 4, "# length of value(payload).");           // NOLINT
DEFINE_uint64(random_seed, 0, "random seed.");
DEFINE_uint64(epoch_duration, 0, "epoch duration in microseconds");
DEFINE_uint64(waiting_resolver_threads, 0, "# waiting resolver threads.");

static bool isReady(const std::vector<char>& readys); // NOLINT
static void waitForReady(const std::vector<char>& readys);

static void invoke_leader();

static void worker(size_t thid, char& ready, const bool& start,
                   const bool& quit, std::vector<Result>& res);

static void invoke_leader() {
    alignas(CACHE_LINE_SIZE) bool start = false;
    alignas(CACHE_LINE_SIZE) bool quit = false;
    alignas(CACHE_LINE_SIZE) std::vector<Result> res(FLAGS_thread); // NOLINT

    std::vector<char> readys(FLAGS_thread); // NOLINT
    std::vector<std::thread> thv;
    for (std::size_t i = 0; i < FLAGS_thread; ++i) {
        thv.emplace_back(worker, i, std::ref(readys[i]), std::ref(start),
                         std::ref(quit), std::ref(res));
    }
    LOG(INFO) << "start waitForReady";
    waitForReady(readys);
    LOG(INFO) << "start ycsb exp.";
    storeRelease(start, true);
#if 0
    for (size_t i = 0; i < FLAGS_duration; ++i) {
        sleepMs(1000);  // NOLINT
    }
#else
    if (sleep(FLAGS_duration) != 0) {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix << "sleep error.";
    }
#endif
    storeRelease(quit, true);
    printf("stop ycsb exp.\n"); // NOLINT
    for (auto& th : thv) th.join();

    for (std::size_t i = 0; i < FLAGS_thread; ++i) {
        res[0].addLocalAllResult(res[i]);
    }
    res[0].displayAllResult(FLAGS_cpumhz, FLAGS_duration, FLAGS_thread);
#if defined(CPR)
    printf("cpr_global_version:\t%zu\n", // NOLINT
           cpr::global_phase_version::get_gpv().get_version());
#endif
    std::cout << "end experiments, start cleanup." << std::endl;
}

static void load_flags() {
    std::cout << "general options" << std::endl;
    if (FLAGS_cpumhz > 1) {
        printf("FLAGS_cpumhz : %zu\n", FLAGS_cpumhz); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "CPU MHz of execution environment. It is used measuring "
                      "some time. It must be larger than 0.";
    }
    if (FLAGS_duration >= 1) {
        printf("FLAGS_duration : %zu\n", FLAGS_duration); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "Duration of benchmark in seconds must be larger than 0.";
    }
    if (FLAGS_key_length > 0) {
        printf("FLAGS_key_length : %zu\n", FLAGS_key_length); // NOLINT
    }
    if (FLAGS_ops >= 1) {
        printf("FLAGS_ops : %zu\n", FLAGS_ops); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "Number of operations in a transaction must be larger "
                      "than 0.";
    }

    // ops_read_type
    printf("FLAGS_ops_read_type : %s\n", // NOLINT
           FLAGS_ops_read_type.data());  // NOLINT
    if (FLAGS_ops_read_type != "point" && FLAGS_ops_read_type != "range") {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix << "Invalid type of read operation.";
    }

    // ops_write_typea
    printf("FLAGS_ops_write_type : %s\n", // NOLINT
           FLAGS_ops_write_type.data());  // NOLINT
    if (FLAGS_ops_write_type != "update" && FLAGS_ops_write_type != "insert" &&
        FLAGS_ops_write_type != "readmodifywrite") {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix << "Invalid type of write operation.";
    }

    if (FLAGS_record > 1) {
        printf("FLAGS_record : %zu\n", FLAGS_record); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1)
                << "Number of database records(tuples) must be large than 0.";
    }
    constexpr std::size_t thousand = 100;
    if (FLAGS_rratio >= 0 && FLAGS_rratio <= thousand) {
        printf("FLAGS_rratio : %zu\n", FLAGS_rratio); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "Rate of reads in a transaction must be in the range 0 "
                      "to 100.";
    }
    if (FLAGS_skew >= 0 && FLAGS_skew < 1) {
        printf("FLAGS_skew : %f\n", FLAGS_skew); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "Access skew of transaction must be in the range 0 to "
                      "0.999... .";
    }
    if (FLAGS_scan_length >= 0) {
      if (FLAGS_scan_length <= FLAGS_record) {
        printf("FLAGS_scan_length : %zu\n", FLAGS_scan_length); // NOLINT
      } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
          << "scan length must be up to the number of records.";
      }
    }
    // LTX options
    if (FLAGS_ltx_ratio >= 0 && FLAGS_ltx_ratio <= 1) {
        printf("FLAGS_ltx_ratio : %f\n", FLAGS_ltx_ratio); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
          << "Access ltx_ratio of transaction must be in the range 0 to 1.";
    }
    if (FLAGS_ltx_ops >= 1) {
        printf("FLAGS_ltx_ops : %zu\n", FLAGS_ltx_ops); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "Number of operations in a transaction must be larger "
                      "than 0.";
    }
    // ltx_read_type
    printf("FLAGS_ltx_read_type : %s\n", // NOLINT
           FLAGS_ltx_read_type.data());  // NOLINT
    if (FLAGS_ltx_read_type != "point" && FLAGS_ltx_read_type != "range" && FLAGS_ltx_read_type != "off") {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix << "Invalid type of read operation.";
    }
    if (FLAGS_ltx_rratio >= 0 && FLAGS_ltx_rratio <= thousand) {
        printf("FLAGS_ltx_rratio : %zu\n", FLAGS_ltx_rratio); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "Rate of reads in a transaction must be in the range 0 "
                      "to 100.";
    }
    // sparse key
    if (FLAGS_sparse_key) {
        printf("FLAGS_sparse_key : true\n"); // NOLINT
    }

    // transaction_type
    printf("FLAGS_transaction_type : %s\n", // NOLINT
           FLAGS_transaction_type.data());  // NOLINT
    if (FLAGS_transaction_type != "short" && FLAGS_transaction_type != "long" &&
        FLAGS_transaction_type != "mix" &&
        FLAGS_transaction_type != "mix_by_thread" && FLAGS_transaction_type != "short_by_thread" &&
        FLAGS_transaction_type != "mix_scanw" && FLAGS_transaction_type != "short_scanw" &&
        FLAGS_transaction_type != "read_only") {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix << "Invalid type of transaction.";
    }

    // about thread
    if (FLAGS_thread >= 1) {
        printf("FLAGS_thread : %zu\n", FLAGS_thread); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "Number of threads must be larger than 0.";
    }

    if (FLAGS_val_length > 1) {
        printf("FLAGS_val_length : %zu\n", FLAGS_val_length); // NOLINT
    } else {
        LOG_FIRST_N(ERROR, 1) << log_location_prefix
                   << "Length of val must be larger than 0.";
    }

    if (!gflags::GetCommandLineFlagInfoOrDie("random_seed").is_default) {
        printf("FLAGS_random_seed : %zu\n", FLAGS_random_seed); // NOLINT
    } else {
        printf("FLAGS_random_seed : (unset)\n"); // NOLINT
    }

    if (!gflags::GetCommandLineFlagInfoOrDie("epoch_duration").is_default) {
        printf("FLAGS_epoch_duration : %zu\n", FLAGS_epoch_duration); // NOLINT
    } else {
        printf("FLAGS_epoch_duration : (unset)\n"); // NOLINT
    }

    if (!gflags::GetCommandLineFlagInfoOrDie("waiting_resolver_threads").is_default) {
        printf("FLAGS_waiting_resolver_threads : %zu\n", FLAGS_waiting_resolver_threads); // NOLINT
    } else {
        printf("FLAGS_waiting_resolver_threads : (unset)\n"); // NOLINT
    }

    printf("Fin load_flags()\n"); // NOLINT
}

int main(int argc, char* argv[]) try { // NOLINT
    google::InitGoogleLogging("shirakami-bench-ycsb");
    gflags::SetUsageMessage(static_cast<const std::string&>(
            "YCSB benchmark for shirakami")); // NOLINT
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_stderrthreshold = 0; // to display info log
    load_flags();

    database_options opt{};
    if (!gflags::GetCommandLineFlagInfoOrDie("epoch_duration").is_default) {
        printf("FLAGS_epoch_duration=%zu\n",FLAGS_epoch_duration);
        opt.set_epoch_time(FLAGS_epoch_duration);
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("waiting_resolver_threads").is_default) {
        printf("FLAGS_waiting_resolver_threads=%zu\n",FLAGS_waiting_resolver_threads);
        opt.set_waiting_resolver_threads(FLAGS_waiting_resolver_threads);
    }
    if (FLAGS_sparse_key) {
        set_key_step(SIZE_MAX / FLAGS_record);
    }
    init(opt); // NOLINT
    LOG(INFO) << "start build_db";
    build_db(FLAGS_record, FLAGS_key_length, FLAGS_val_length, FLAGS_thread);
    LOG(INFO) << "start invoke_leader";
    invoke_leader();
    fin();

    return 0;
} catch (std::exception& e) { std::cerr << e.what() << std::endl; }

bool isReady(const std::vector<char>& readys) { // NOLINT
    for (const char& b : readys) {              // NOLINT
        if (loadAcquire(b) == 0) return false;
    }
    return true;
}

void waitForReady(const std::vector<char>& readys) {
    while (!isReady(readys)) { _mm_pause(); }
}

void worker(const std::size_t thid, char& ready, const bool& start,
            const bool& quit, std::vector<Result>& res) {
    // init work

    Xoroshiro128Plus rnd;
    if (!gflags::GetCommandLineFlagInfoOrDie("random_seed").is_default) {
        rnd.seed(FLAGS_random_seed + thid);
    }
    size_t key_max = FLAGS_record;
    size_t key_step = 1;
    if (FLAGS_sparse_key) {
        key_step = SIZE_MAX / FLAGS_record;
        key_max = key_step * FLAGS_record;
    }
    FastZipf zipf(&rnd, FLAGS_skew, FLAGS_record, key_step);
    std::reference_wrapper<Result> myres = std::ref(res[thid]);

    // this function can be used in Linux environment only.
#ifdef SHIRAKAMI_LINUX
    setThreadAffinity(static_cast<const int>(thid));
#endif

    Token token{};
    std::vector<shirakami::opr_obj> opr_set;
    opr_set.reserve(FLAGS_ltx_ops > FLAGS_ops ? FLAGS_ltx_ops : FLAGS_ops);
    auto ret = enter(token);
    if (ret != Status::OK) { LOG(FATAL) << "too many tx handle: " << ret; }
    auto* ti = static_cast<session*>(token);

    uint64_t ltx_ratio = static_cast<uint64_t>(FLAGS_ltx_ratio * UINT64_MAX);
    uint64_t ltx_commit_counts = 0;
    bool wo_wp = FLAGS_rratio==100 && (FLAGS_ltx_ratio==0 || FLAGS_ltx_rratio==100);
    size_t retry_scan_count=0;
    size_t retry_update_count=0;
    size_t retry_insert_count=0;
    size_t insert_count=0;
    size_t tx_count=0;
    storeRelease(ready, 1);
    while (!loadAcquire(start)) _mm_pause();
    while (likely(!loadAcquire(quit))) {
        //if (++tx_count>20) goto leaving;
        bool is_ltx;
        if (FLAGS_transaction_type == "mix_by_thread" || FLAGS_transaction_type == "short_by_thread") {
            if (thid!=0) {
                // gen query contents
                gen_tx_rw(opr_set, FLAGS_key_length, key_max, FLAGS_thread, thid,
                          FLAGS_ops, FLAGS_ops_read_type, FLAGS_ops_write_type,
                          FLAGS_rratio, rnd, zipf);
                is_ltx = false;
            } else {
                // gen query contents
                gen_tx_scan(opr_set, FLAGS_key_length, key_max,
                            FLAGS_scan_length*key_step, rnd, zipf);
                is_ltx = true;
            }
        } else if (FLAGS_transaction_type == "mix_scanw" || FLAGS_transaction_type == "short_scanw") {
            // gen scan
            gen_tx_scan(opr_set, FLAGS_key_length, key_max,
                        FLAGS_scan_length*key_step, rnd, zipf);
            // gen update
            for (std::size_t i = 0; i < FLAGS_ops; ++i) {
                std::uint64_t keynm = zipf() % key_max;
                opr_set.emplace_back(OP_TYPE::UPDATE,
                    make_key(FLAGS_key_length, keynm)); // NOLINT
            }
            is_ltx = true;
        } else {
            if (rnd.next() < ltx_ratio) {
                std::string read_type;
                read_type = (FLAGS_ltx_read_type == "off") ?
                  FLAGS_ops_read_type : FLAGS_ltx_read_type;
                // gen query contents
                gen_tx_rw(opr_set, FLAGS_key_length, key_max, FLAGS_thread, thid,
                          FLAGS_ltx_ops, read_type, FLAGS_ops_write_type,
                          FLAGS_ltx_rratio, FLAGS_scan_length * key_step, rnd, zipf);
                is_ltx = true;
            } else {
                // gen query contents
                gen_tx_rw(opr_set, FLAGS_key_length, key_max, FLAGS_thread, thid,
                          FLAGS_ops, FLAGS_ops_read_type, FLAGS_ops_write_type,
                          FLAGS_rratio, FLAGS_scan_length * key_step, rnd, zipf);
                is_ltx = false;
            }
        }

        if (ret == Status::WARN_ALREADY_BEGIN) { LOG(FATAL); }

        // tx begin
        transaction_options::transaction_type tt{};
        if (FLAGS_transaction_type == "short" || FLAGS_transaction_type == "short_by_thread" ||
            (FLAGS_transaction_type == "mix_by_thread" && thid!=0) ||
            FLAGS_transaction_type == "short_scanw") {
            tt = transaction_options::transaction_type::SHORT;
            ret = tx_begin({token, tt});
        } else if (FLAGS_transaction_type == "long" || FLAGS_transaction_type == "mix" ||
                   (FLAGS_transaction_type == "mix_by_thread" && thid==0) ||
                   FLAGS_transaction_type == "mix_scanw") {
            tt = (FLAGS_transaction_type == "long" || is_ltx) ?
                 transaction_options::transaction_type::LONG :
                 transaction_options::transaction_type::SHORT;
            if (wo_wp || !is_ltx) { // NOLINT
                ret = tx_begin({token, tt});
            } else {
                ret = tx_begin({token, tt, {storage}});
            }
            // wait start epoch
            auto* ti = static_cast<session*>(token);
            while (epoch::get_global_epoch() < ti->get_valid_epoch()) {
                _mm_pause();
            }
        } else if (FLAGS_transaction_type == "read_only") {
            tt = transaction_options::transaction_type::READ_ONLY;
            ret = tx_begin({token, tt});
        } else {
            LOG(FATAL) << log_location_prefix << "invalid transaction type";
        }
        if (ret != Status::OK) {
            LOG(FATAL) << log_location_prefix << "unexpected error. " << ret << " is_ltx=" << (is_ltx?"t":"f") << " wo_wp=" << (wo_wp?"t":"f") << " tt=" << tt << " ";
        }

        // execute operations
        //size_t itr_count=0;
        for (auto&& itr : opr_set) {
            //printf("%zu|%zu|%i|%s\n", thid, itr_count, itr.get_type(), itr.get_key());
            /*
            std::cout << thid << "|" <<tx_count<< "|" <<itr_count<< "|" <<itr.get_type()<< "|";
            if (itr.get_type() == OP_TYPE::SCAN) {
              for (char c : itr.get_scan_l_key()) {
                std::cout << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c)) << " ";
              }
              std::cout << "|";
              for (char c : itr.get_scan_r_key()) {
                std::cout << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c)) << " ";
              }
            } else {
              for (char c : itr.get_key()) {
                // unsigned char でキャストすることで、バイト値を正しく扱います [1]。
                // static_cast<unsigned char>(c) とすることで、正しく16進数で表示されます [1]。
                std::cout << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c)) << " ";
              }
            }
            std::cout << "|" <<std::endl;
            ++itr_count;
             */
            if (itr.get_type() == OP_TYPE::SEARCH) {
                for (;;) {
                    std::string vb{};
                    ret = search_key(token, storage, itr.get_key(), vb);
                    if (ret == Status::OK) { break; }
                    if (ret == Status::ERR_CC) { goto ABORTED; } // NOLINT
                    if (ret == Status::ERR_FATAL) {
                        LOG(FATAL) << log_location_prefix;
                    }
                    if (loadAcquire(quit)) goto ABORT_WITHOUT_COUNT;
                }
            } else if (itr.get_type() == OP_TYPE::UPDATE) {
            retry_update:
                ret = update(token, storage, itr.get_key(),
                             std::string(FLAGS_val_length, '0'));
                if (ret == Status::WARN_WRITE_WITHOUT_WP || ret == Status::WARN_CONFLICT_ON_WRITE_PRESERVE) {
                    ++retry_update_count;
                    if (loadAcquire(quit)) goto ABORT_WITHOUT_COUNT;
                    _mm_pause();
                    goto retry_update;
                    //goto ABORTED;
                }
                if (ret != Status::OK || ret == Status::ERR_CC) {
                    LOG(FATAL) << "unexpected error, rc: " << ret << " ";
                }
            } else if (itr.get_type() == OP_TYPE::INSERT) {
            retry_insert:
                insert_count++;
                ret = insert(token, storage, itr.get_key(),
                             std::string(FLAGS_val_length, '0'));
                if (ret == Status::WARN_WRITE_WITHOUT_WP || ret == Status::WARN_CONFLICT_ON_WRITE_PRESERVE) {
                    ++retry_insert_count;
                    if (loadAcquire(quit)) goto ABORT_WITHOUT_COUNT;
                    _mm_pause();
                    goto retry_insert;
                    //goto ABORTED;
                }
                if (ret == Status::WARN_ALREADY_EXISTS) {
                    goto ABORTED;
                }
                if (ret != Status::OK) {
                    LOG(FATAL) << "unexpected error, rc: " << ret << " is_ltx=" << is_ltx << " ";
                }
                // rarely, ret == already_exist due to design
            } else if (itr.get_type() == OP_TYPE::SCAN) {
                ScanHandle hd{};
            retry_scan:
                ret = open_scan(token, storage, itr.get_scan_l_key(),
                                scan_endpoint::INCLUSIVE, itr.get_scan_r_key(),
                                scan_endpoint::INCLUSIVE, hd,
                                FLAGS_scan_length);
                if (ret == Status::WARN_PREMATURE) {
                    ++retry_scan_count;
                    if (loadAcquire(quit)) goto ABORT_WITHOUT_COUNT;
                    _mm_pause();
                    goto retry_scan;
                    //goto ABORTED;
                }
                if (ret == Status::ERR_CC) { goto ABORTED; } // NOLINT
                if (ret != Status::OK || ret == Status::ERR_CC) {
                    LOG(FATAL) << "unexpected error, rc: " << ret << " ";
                }
                std::string vb{};
                size_t read_count=0, warn_count=0;
                do {
                    do {
                        ret = read_value_from_scan(token, hd, vb);
                        if (loadAcquire(quit)) goto ABORT_WITHOUT_COUNT;
                        _mm_pause();
                    } while (ret == Status::WARN_CONCURRENT_INSERT || ret == Status::WARN_CONCURRENT_UPDATE);
                    if (ret == Status::ERR_CC) { goto ABORTED; } // NOLINT
                    if (ret == Status::WARN_NOT_FOUND) {
                      goto read_next;
                      /*
                      if (++warn_count<10) {continue;}
                      uint64_t lkey, rkey;
                      std::memcpy(&lkey, itr.get_scan_l_key().data(), sizeof(lkey));
                      lkey=__builtin_bswap64(lkey);
                      std::memcpy(&rkey, itr.get_scan_r_key().data(), sizeof(rkey));
                      rkey=__builtin_bswap64(rkey);
                      printf("lkey=%zu, lkey_pos=%f\nrkey=%zu, key_step=%zu, (rkey-lkey+1)/key_step=%zu, read_count=%zu\n",
                        lkey, lkey*1.0/key_max, rkey, key_step, (rkey-lkey+1)/key_step, read_count);
                      goto ABORTED;
                      */
                    }
                    if (ret != Status::OK) { LOG(FATAL) << "unexpected error: " << ret << " "; }
                    ++read_count;
                read_next:
                    ret = next(token, hd);
                    if (loadAcquire(quit)) {
                        // for fast exit if it is over exp time.
                        // for scan loop
                        goto ABORT_WITHOUT_COUNT;
                    }
                } while (ret != Status::WARN_SCAN_LIMIT);
                ret = close_scan(token, hd);
                if (ret != Status::OK) { LOG(FATAL) << "unexpected error: " << ret << " "; }
            }
            if (loadAcquire(quit)) {
                // for fast exit if it is over exp time.
                // for operation loop
                goto ABORT_WITHOUT_COUNT;
            }
        }

    RETRY_COMMIT:
        ret = commit(token);
        if (ret == Status::WARN_WAITING_FOR_OTHER_TX) {
            // ltx
            do {
                _mm_pause();
                ret = check_commit(token);
                if (loadAcquire(quit)) {
                    // for fast exit if it is over exp time.
                    // for waiting commit

                    // should goto ABORT_WITHOUT_COUNT,
                    // but aborting after commit request is not stable
                    // so leave the transaction as it is.
                    goto leaving;
                    return;
                }
            } while (ret == Status::WARN_WAITING_FOR_OTHER_TX);
        }
        if (ret == Status::OK) { // NOLINT
            ++myres.get().get_local_commit_counts();
            if (is_ltx) ++myres.get().get_local_ltx_commit_counts();
        } else {
    ABORTED: // NOLINT
            ++myres.get().get_local_abort_counts();
            if (is_ltx) ++myres.get().get_local_ltx_abort_counts();
    ABORT_WITHOUT_COUNT: // NOLINT
            abort(token);
        }
    }
    ret = leave(token);
    if (ret != Status::OK) { LOG_FIRST_N(ERROR, 1) << ret; }
 leaving:
    if (retry_scan_count>0) printf("retry_scan_count=%zu thid=%zu\n",retry_scan_count,thid);
    if (retry_update_count>0) printf("retry_update_count=%zu thid=%zu\n",retry_update_count,thid);
    if (retry_insert_count>0) printf("retry_insert_count=%zu thid=%zu\n",retry_insert_count,thid);
    if (insert_count>0) printf("insert_count=%zu thid=%zu\n",insert_count,thid);
}
