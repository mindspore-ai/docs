cmake_minimum_required(VERSION 3.14.1)
project (TensorAddSample[CXX])
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIE -Wl,--allow-shlib-undefined")
option(MINDSPORE_PATH "mindspore install path" "")
include_directories(${MINDSPORE_PATH})
find_library(MS_LIB libmindspore.so ${MINDSPORE_PATH}/lib)

add_executable(tensor_add_sample main.cc)
target_link_libraries(tensor_add_sample ${MS_LIB})
