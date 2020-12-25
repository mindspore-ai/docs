# 下载MindSpore Lite

`Windows` `Linux` `Android` `环境准备` `初级` `中级` `高级`

<!-- TOC -->

- [下载MindSpore Lite](#下载mindspore-lite)
    - [1.0.1](#101)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_zh_cn/use/downloads.md" target="_blank"><img src="../_static/logo_source.png"></a>

欢迎使用MindSpore Lite，我们提供了支持多种操作系统和硬件平台的模型转换、模型推理、图像处理等功能，你可以下载适用于本地环境的版本包直接使用。

## 1.0.1

|   组件   |   硬件平台   |   操作系统   |      链接      |        SHA-256     |
|    ---   |    ---   |    ---   |       ---      |    ---      |
|   MindSpore Lite 模型转换工具（Converter）   |   CPU   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/lite/ubuntu_x86/mindspore-lite-1.0.1-converter-ubuntu.tar.gz> |9498d721645e97992b7d5a46246d42db31114952d00bdecc0c40510cb629347e   |
|      |      |   Windows-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/lite/windows_x86/mindspore-lite-1.0.1-converter-win-cpu.zip>   |2040d2a71a90ffabca108ef3195a2fb3cbef07b73ef2197bb63097fba2ac6a33   |
|   MindSpore Lite 模型推理框架（Runtime，含图像处理）   |   CPU   |   Android-aarch32   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/lite/android_aarch32/mindspore-lite-1.0.1-runtime-arm32-cpu.tar.gz>   |3c99c47efbf0df16d8627b14c3da8d80a13f246ee409b10edbcde6b9d0bc4261   |
|      |      |   Android-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/lite/android_aarch64/mindspore-lite-1.0.1-runtime-arm64-cpu.tar.gz>   |4306b5b2ecb7324133eab27f40c6c05efa1be28b2e2ecd1c35b152ef15de5482   |
|      |   GPU   |   Android-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/lite/android_aarch64/mindspore-lite-1.0.1-runtime-arm64-gpu.tar.gz>   |09407dff8cc0aee5a8075a12a4fbde10634aafde238eeb686c3cf91481c667b5   |

MindSpore Lite还提供对Runtime的`libmindspore-lite.a`[静态库裁剪工具](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/cropper_tool.html#)，用于剪裁静态库文件，有效降低库文件大小。
