cmake_minimum_required(VERSION 3.14)
project(nnie_proposal)

set(MSLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_patry/")
execute_process(
        COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
        OUTPUT_VARIABLE  result
        WORKING_DIRECTORY ${MSLIB_DIR})
execute_process(
        COMMAND ${CMAKE_COMMAND} --build .
        WORKING_DIRECTORY ${MSLIB_DIR})

#execute_process(COMMAND tar xzf  "ms-prefix/src/mindspore-lite-1.0.1-runtime-arm64-cpu.tar.gz" WORKING_DIRECTORY ${MSLIB_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_patry/ms-prefix/src/ms/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_patry/ms-prefix/src/third_patry/flatbuffers/include)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC)

add_library(nnie_proposal SHARED
        ${SRC})
