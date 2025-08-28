file(GLOB YCSB_LTX_SOURCES
        "ycsb_ltx.cpp"
        "build_db.cpp"
        "result_ltx.cpp"
        )

add_executable(ycsb_ltx
        ${YCSB_LTX_SOURCES}
        )

target_link_libraries(ycsb_ltx
        PRIVATE gflags::gflags
        PRIVATE glog::glog
        PRIVATE shirakami
        PRIVATE shirakami-impl
        PRIVATE ${tbb_prefix}tbb
        PRIVATE ${tbb_prefix}tbbmalloc
        PRIVATE ${tbb_prefix}tbbmalloc_proxy
        PRIVATE Boost::filesystem
        PRIVATE Threads::Threads
        PRIVATE yakushima
        PRIVATE atomic
        )

if (BUILD_PWAL)
  target_link_libraries(ycsb_ltx
        PRIVATE limestone
  )
endif()

target_include_directories(ycsb_ltx
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${PROJECT_SOURCE_DIR}/bench
        PRIVATE ${PROJECT_SOURCE_DIR}/bench/include
        PRIVATE ${PROJECT_SOURCE_DIR}/include
        PRIVATE ${PROJECT_SOURCE_DIR}/src
        PRIVATE ${PROJECT_SOURCE_DIR}/src/include
        PRIVATE ${PROJECT_SOURCE_DIR}/test/include
        PRIVATE ${gflags_INCLUDE_DIR}
        )

add_test(
        NAME ycsb_ltx
        COMMAND ycsb_ltx)
