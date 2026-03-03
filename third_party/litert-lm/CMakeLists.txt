# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


if(CMAKE_VERSION VERSION_LESS "3.25")
    message(FATAL_ERROR
        "CMake 3.25+ is required because we use the 'block()' command for variable isolation.\n"
        "Your current version is: ${CMAKE_VERSION}.\n"
        "Please upgrade CMake to proceed."
    )
endif()

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(LiteRT-LM LANGUAGES C CXX)


# --- Global Paths
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(LITERTLM_ROOT_DIR "${PROJECT_ROOT}" CACHE PATH "LiteRT-LM: Root of the LiteRT-LM workspace")
set(LITERTLM_PACKAGES_DIR "${LITERTLM_ROOT_DIR}/cmake/packages" CACHE PATH "LiteRT-LM: Directory for build packages")
set(LITERTLM_PATCHES_DIR "${LITERTLM_ROOT_DIR}/cmake/patches"
    CACHE PATH "LiteRT-LM: Directory for build patches")
set(LITERTLM_MODULES_DIR "${LITERTLM_ROOT_DIR}/cmake/modules"
    CACHE PATH "LiteRT-LM: Modules path, appended to CMAKE_MODULE_PATH")
set(LITERTLM_SCRIPTS_DIR "${LITERTLM_ROOT_DIR}/cmake/scripts"
    CACHE PATH "LiteRT-LM: Directory for internal build and automation scripts")
set(LITERTLM_LOCAL_STAGING_DIR "${CMAKE_BINARY_DIR}/staging/lib" CACHE INTERNAL "")
file(MAKE_DIRECTORY "${LITERTLM_LOCAL_STAGING_DIR}")
set(LITERTLM_RUST_TARGET "" CACHE STRING "Rust target triple (e.g., aarch64-linux-android). Leave empty for host.")
set(LITERTLM_RUST_FILES
    "${PROJECT_ROOT}/runtime/components/rust/minijinja_template.rs"
    "${PROJECT_ROOT}/runtime/components/tool_use/rust/parsers.rs"
    CACHE INTERNAL "Rust files to include in the cxx bridge")
set(LITERTLM_CARGO_TOML "${PROJECT_ROOT}/Cargo.toml" CACHE INTERNAL "Path to LiteRT-LM's Cargo.toml")
set(LITERTLM_PROTO_FILES
  "${PROJECT_ROOT}/runtime/proto/engine.proto"
  "${PROJECT_ROOT}/runtime/proto/llm_metadata.proto"
  "${PROJECT_ROOT}/runtime/proto/llm_model_type.proto"
  "${PROJECT_ROOT}/runtime/proto/sampler_params.proto"
  "${PROJECT_ROOT}/runtime/proto/token.proto"
  "${PROJECT_ROOT}/runtime/executor/proto/constrained_decoding_options.proto"
  "${PROJECT_ROOT}/runtime/util/external_file.proto"
  CACHE INTERNAL
  "Protobuf files that will be processed with protoc"
)
set(LITERTLM_FLATBUFFERS_FILES
    "${PROJECT_ROOT}/schema/litertlm_header_schema.fbs"
    CACHE INTERNAL
    "FlatBuffers files that will be processed with flatc"
)

### TODO(totero): Replace references to these with the namespaced versions.
set(SCRIPTS_DIR      "${LITERTLM_SCRIPTS_DIR}")
set(PATCHES_DIR      "${LITERTLM_PATCHES_DIR}")
set(MODULES_DIR        "${LITERTLM_MODULES_DIR}")
list(APPEND CMAKE_MODULE_PATH   "${LITERTLM_MODULES_DIR}")
###

set(EXTERNAL_PROJECT_BINARY_DIR "${CMAKE_BINARY_DIR}/external")
set(THIRD_PARTY_DIR             "${CMAKE_BINARY_DIR}/third_party")
set(GENERATED_SRC_DIR           "${CMAKE_BINARY_DIR}/generated/src" CACHE PATH "Root for generated code")
set(LITERTLM_INCLUDE_PATHS
  "${GENERATED_SRC_DIR}"
  "${CMAKE_BINARY_DIR}"
  CACHE INTERNAL ""
)

option(_unverified_targets "Builds executables that are still in progress" OFF)


# --- Environment
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.20 CACHE STRING "Required CMake policy version." FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive" CACHE STRING "Required CXX flags" FORCE)

set_property(GLOBAL PROPERTY LITERTLM_LOCAL_ARCHIVE_REGISTRY "")
set_property(GLOBAL PROPERTY LITERTLM_LOCAL_TARGET_REGISTRY "")

# --- Optimization
option(ENABLE_CCACHE "Enables the use of ccache" OFF)
if(${ENABLE_CCACHE} STREQUAL "ON")
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
      message(STATUS "🚀 Using CCache to speed up builds")
      set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  else()
      message(WARNING "⚠️ CCache not found. Install it (`sudo apt install ccache`) for faster builds!")
  endif()
endif()

# --- Utilities
include("${LITERTLM_MODULES_DIR}/utils.cmake")
include(macros)

# --- Dependencies
include("${LITERTLM_MODULES_DIR}/fetch_content.cmake")
include("${LITERTLM_MODULES_DIR}/external_project.cmake")
include("${LITERTLM_MODULES_DIR}/collect_dependencies.cmake")

if(TARGET LITERTLM_DEPS)
    message(DEBUG "✅ DEBUG: LITERTLM_DEPS exists!")
else()
    message(FATAL_ERROR "❌ DEBUG: LITERTLM_DEPS is MISSING!")
endif()


# --- LiteRT-LM Targets
include("${LITERTLM_MODULES_DIR}/generators.cmake")
generate_src_files(LITERTLM_CORE_FILES ${ALL_SOURCE_FILES} ${ALL_HEADER_FILES})
generate_src_files(LITERTLM_FLATBUFFER_FILES ${ALL_SCHEMA_FILES})

if(LITERTLM_CORE_FILES)
    include("${LITERTLM_MODULES_DIR}/patcher.cmake")
    set_property(
        DIRECTORY
        APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${LITERTLM_PATCHES_DIR}/litertlm_patches.json"
    )
    apply_patches_from_json("${LITERTLM_PATCHES_DIR}/litertlm_patches.json")

    if(NOT EXISTS "${GENERATED_SRC_DIR}/runtime/components/constrained_decoding/gemma_model_constraint_provider.cc")
      file(COPY "${LITERTLM_PATCHES_DIR}/stubs/gemma_model_constraint_provider.cc"
        DESTINATION "${GENERATED_SRC_DIR}/runtime/components/constrained_decoding"
      )
    endif()
endif()

add_compile_definitions(
    absl_nullable=
    absl_nonnull=
)

set(TENSORFLOW_MISSING_SRCS
    "${TENSORFLOW_SOURCE_DIR}/tensorflow/compiler/mlir/lite/allocation.cc"
    "${TENSORFLOW_SOURCE_DIR}/tensorflow/compiler/mlir/lite/mmap_allocation.cc"
    "${TENSORFLOW_SOURCE_DIR}/tensorflow/compiler/mlir/lite/mmap_allocation_disabled.cc"
    "${TENSORFLOW_SOURCE_DIR}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc"
    "${TENSORFLOW_SOURCE_DIR}/tensorflow/compiler/mlir/lite/core/model_builder_base.cc"

)
set_source_files_properties(${TENSORFLOW_MISSING_SRCS} PROPERTIES GENERATED TRUE)

add_subdirectory(c)
add_subdirectory(schema)
add_subdirectory(runtime)

include("${LITERTLM_MODULES_DIR}/local_aggregate.cmake")
generate_local_aggregate()

add_executable(litert_lm_main
    "runtime/engine/litert_lm_main.cc"
)

if(TARGET litertlm_local_anchor)
    add_dependencies(litert_lm_main litertlm_local_anchor)
endif()

target_link_options(litert_lm_main PRIVATE
  "LINKER:--export-dynamic-symbol=LiteRt*"
)

target_compile_definitions(litert_lm_main PRIVATE
    ENABLE_HUGGINGFACE_TOKENIZER
    ENABLE_SENTENCEPIECE_TOKENIZER
)

separate_arguments(_LITERTLM_ODML_EXPANDED_LIST NATIVE_COMMAND "${_LITERTLM_ODML_PAYLOAD}")
separate_arguments(_LITERTLM_CORE_EXPANDED_LIST NATIVE_COMMAND "${_LITERTLM_CORE_PAYLOAD}")
target_link_libraries(litert_lm_main
    PUBLIC
        "-Wl,--allow-multiple-definition"
        "-Wl,--start-group"
        "-Wl,--whole-archive"
        ${_LITERTLM_ODML_EXPANDED_LIST}
        "-Wl,--no-whole-archive"
        ${_LITERTLM_CORE_EXPANDED_LIST}
        libpng_lib
        kissfft_lib
        miniaudio_lib
        minizip_lib
        minja_lib
        zlib_lib
        antlr_lib
        LiteRTLM::nlohmann_json::nlohmann_json
        opencl_headers_lib
        "-lz" "-lrt" "-lpthread" "-ldl"
        "-Wl,--end-group"
)

target_include_directories(litert_lm_main PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)
