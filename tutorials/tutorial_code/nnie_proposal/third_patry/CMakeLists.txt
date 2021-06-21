cmake_minimum_required(VERSION 3.14)
project(ms_download)

include(ExternalProject)
set(MS_VESION "1.3.0")
message(STATUS "download mindspore-lite-${MS_VESION}-runtime-arm64-cpu")

SET(MS_URL https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VESION}/lite/android_aarch64/mindspore-lite-${MS_VESION}-runtime-arm64-cpu.tar.gz)

ExternalProject_Add(
        ms
        URL ${MS_URL}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        UPDATE_COMMAND "" # Skip annoying updates for every build
        # Disable install step
        INSTALL_COMMAND ""
)

add_custom_target(ms_download ALL
        DEPENDS ms
        )
