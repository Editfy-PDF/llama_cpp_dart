cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME "llamalib")
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES C)
cmake_policy(VERSION 3.14...3.25)

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

if(NOT EXISTS ${LLAMA_DIR})
  message(STATUS "Downloading llama.cpp...")
  execute_process(COMMAND git clone --recursive https://github.com/ggml-org/llama.cpp ${LLAMA_DIR})
endif()

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs" FORCE)

if(ANDROID)
    set(GGML_OPENMP OFF CACHE BOOL "ggml: disable -march=native flag" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(${LLAMA_DIR})

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND ANDROID)
    target_link_options(llama PRIVATE "-Wl,-z,max-page-size=16384")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
endif()

target_compile_definitions(llama PUBLIC DART_SHARED_LIB)