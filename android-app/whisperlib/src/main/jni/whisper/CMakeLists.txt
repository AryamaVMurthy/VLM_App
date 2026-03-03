cmake_minimum_required(VERSION 3.10)

project(whisper.cpp)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED WHISPER_LIB_DIR)
    message(FATAL_ERROR "WHISPER_LIB_DIR is not set. Pass -DWHISPER_LIB_DIR=<absolute whisper.cpp path> from Gradle.")
endif()

if(NOT EXISTS "${WHISPER_LIB_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "WHISPER_LIB_DIR='${WHISPER_LIB_DIR}' is invalid. Expected whisper.cpp root with CMakeLists.txt.")
endif()

# Get whisper.cpp version
file(READ "${WHISPER_LIB_DIR}/CMakeLists.txt" MAIN_CMAKE_CONTENT)
string(REGEX MATCH "project\\(\"whisper\\.cpp\" VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)\\)" VERSION_MATCH "${MAIN_CMAKE_CONTENT}")
if(CMAKE_MATCH_1)
    set(WHISPER_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
else()
    set(WHISPER_VERSION "unknown" PARENT_SCOPE)
endif()

message(STATUS " Whisper version: ${WHISPER_VERSION}")

set(
    SOURCE_FILES
    ${WHISPER_LIB_DIR}/src/whisper.cpp
    ${CMAKE_SOURCE_DIR}/jni.c
    )

find_library(LOG_LIB log)

include(FetchContent)

function(build_library target_name)
    set(WHISPER_CPU_ARCH_FLAGS)
    set(WHISPER_NATIVE_OPT_FLAGS -O3 -ffast-math -fno-finite-math-only -fomit-frame-pointer -flto -DNDEBUG)

    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
    )

    target_compile_definitions(${target_name} PUBLIC GGML_USE_CPU)
    target_compile_definitions(${target_name} PRIVATE WHISPER_VERSION="${WHISPER_VERSION}")

    if (${target_name} STREQUAL "whisper_v8fp16_va")
        target_compile_options(${target_name} PRIVATE -march=armv8.2-a+fp16)
        set(WHISPER_CPU_ARCH_FLAGS -march=armv8.2-a+fp16)
    elseif (${target_name} STREQUAL "whisper_vfpv4")
        target_compile_options(${target_name} PRIVATE -mfpu=neon-vfpv4)
        set(WHISPER_CPU_ARCH_FLAGS -mfpu=neon-vfpv4)
    endif ()

    # Keep debug symbols from Gradle's Debug build type while forcing optimized native kernels.
    target_compile_options(${target_name} PRIVATE ${WHISPER_NATIVE_OPT_FLAGS})
    target_link_options(${target_name} PRIVATE -flto)

    if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
        target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)
        target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
        target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
    endif ()

    FetchContent_Declare(ggml SOURCE_DIR ${WHISPER_LIB_DIR}/ggml)
    FetchContent_MakeAvailable(ggml)
    foreach(GGML_TARGET IN ITEMS ggml ggml-base ggml-cpu)
        if(TARGET ${GGML_TARGET})
            target_compile_options(${GGML_TARGET} PRIVATE ${WHISPER_CPU_ARCH_FLAGS} ${WHISPER_NATIVE_OPT_FLAGS})
            target_link_options(${GGML_TARGET} PRIVATE -flto)
        endif()
    endforeach()
    target_link_libraries(${target_name} ${LOG_LIB} android ggml)


endfunction()

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("whisper_v8fp16_va")
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library("whisper_vfpv4")
endif ()

build_library("whisper") # Default target

include_directories(${WHISPER_LIB_DIR})
include_directories(${WHISPER_LIB_DIR}/src)
include_directories(${WHISPER_LIB_DIR}/include)
include_directories(${WHISPER_LIB_DIR}/ggml/include)
include_directories(${WHISPER_LIB_DIR}/ggml/src)
include_directories(${WHISPER_LIB_DIR}/ggml/src/ggml-cpu)
