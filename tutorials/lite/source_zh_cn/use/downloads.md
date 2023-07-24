# 下载MindSpore Lite

`Windows` `Linux` `Android` `环境准备` `初级` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_zh_cn/use/downloads.md)

欢迎使用MindSpore Lite，我们提供了支持多种操作系统和硬件平台的模型转换、模型推理、图像处理等功能，你可以下载适用于本地环境的版本包直接使用。

## 1.1.0

### 推理

|   组件   |   硬件平台   |   操作系统   |   链接   |   SHA-256   |
|   ---   |     ---     |     ---     |   ---   |     ---     |
| MindSpore Lite 模型转换工具（Converter）         | CPU         | Ubuntu-x64                      | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/linux/mindspore-lite-1.1.0-converter-linux-x64.tar.gz> | d449e38a8493c314d1b5b1a127f62269192da785b012ff892eda775dedca3d82 |
|                                               | CPU         | Windows-x64                     | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/windows/mindspore-lite-1.1.0-converter-win-x64.zip> | 5e50b7701b97ebe784095f2ba954fc6c377eb157fbc9aaeae2497e38cc4ee212 |
| MindSpore Lite 模型推理框架（Runtime，含图像处理） | CPU/GPU/NPU | Android-aarch64/Android-aarch32 | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/android/mindspore-lite-1.1.0-inference-android.tar.gz> | a19de5706db57e97a5f04ef08e0e383f8ea497c70bb60e60d056b31a603c0243 |
|                                               | CPU         |  Ubuntu-x64                     | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/linux/mindspore-lite-1.1.0-inference-linux-x64.tar.gz> | 176256c2fbef775f1a44aaeccae0c4eea6a60f41fc0baece5479dcb378155f36 |
|                                               | CPU         |  Windows-x64                    | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/windows/mindspore-lite-1.1.0-inference-win-x64.zip> | 30b5545245832a73d84732166f360c77cd09a7a4fe1fb922a8f7b80e7df326c1 |

### 训练

|   组件   |   硬件平台   |   操作系统   |   链接   |   SHA-256   |
|   ---   |     ---     |     ---     |   ---   |     ---     |
| MindSpore Lite 模型转换工具（Converter）         | CPU         | Ubuntu-x64                      | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/linux/mindspore-lite-1.1.0-train-converter-linux-x64.tar.gz> | f95a9db98c84ec3d97f88383ecc3832582aa9737ed287c33703deb0b419acf25 |
| MindSpore Lite 模型训练框架（Runtime，含图像处理） | CPU | Android-aarch64/Android-aarch32 | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/android/mindspore-lite-1.1.0-train-android.tar.gz> | a6d8152f4e2d674c52af2c379f7d07858d30bc0dceef1dbc366e6fa16a5948b5 |
|                                               | CPU         |  Ubuntu-x64                     | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/linux/mindspore-lite-1.1.0-train-linux-x64.tar.gz> | 1290f0adc790adc9edce654b9a629a9a323cfcb8453eb6bc19b779ef726282bf |

Android-aarch32不支持GPU和NPU。

MindSpore Lite还提供对Runtime的`libmindspore-lite.a`[静态库裁剪工具](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/cropper_tool.html#)，用于剪裁静态库文件，有效降低库文件大小。
