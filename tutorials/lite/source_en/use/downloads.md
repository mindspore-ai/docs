# Downloading MindSpore Lite

`Windows` `Linux` `Android` `Environment Preparation` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/tutorials/lite/source_en/use/downloads.md)

Welcome to MindSpore Lite. We provide functions such as model conversion, model inference, image processing, etc. that support multiple operating systems and hardware platforms. You can download the version package suitable for the local environment and use it directly.

## 1.2.0

|  Module Name  | Hardware Platform |  Operating System  | Download Links |   SHA-256   |
|      ---      |       ---         |         ---        |      ---       |    ---      |
| Inference runtime (cpp), training runtime (cpp), inference aar package, and benchmark/benchmark_train tools. | CPU     | Android-aarch32 | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.0/MindSpore/lite/release/android/mindspore-lite-1.2.0-android-aarch32.tar.gz> | 7d073573385a69bff53542c395d106393da241682cd6053703ce21f1de23bac6 |
| Inference runtime (cpp), training runtime (cpp), inference aar package, and benchmark/benchmark_train tools. | CPU/GPU | Android-aarch64 | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.0/MindSpore/lite/release/android/gpu/mindspore-lite-1.2.0-android-aarch64.tar.gz> | 7f8400f0b97fa3e7cbf0d266c73b43a2410905244b04d0202fab39d9267346e0 |
| Inference runtime (cpp), training runtime (cpp), inference jar package, and benchmark/benchmark_train/codegen/converter/cropper tools. | CPU     |  Ubuntu-x64     | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.0/MindSpore/lite/release/linux/mindspore-lite-1.2.0-linux-x64.tar.gz> | 3b609ed8be9e3ae70987d6e00421ad4720776d797133e72f6952ba6b93059062 |
| Inference runtime (cpp) and benchmark/codegen/converter tools. | CPU     |  Windows-x64    | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.0/MindSpore/lite/release/windows/mindspore-lite-1.2.0-win-x64.zip> | bf01851d7e2cde416502dce11bd2a86ef63e559f6dabba090405755a87ce14ae |
| Inference runtime(cpp) | CPU     |  OpenHarmony    | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.0/MindSpore/lite/release/openharmony/mindspore-lite-1.2.0-ohos-aarch32.tar.gz> | a9987b25815cb69e0f630be1388486e8d727a19815a67851089b7d633bd2f3f2 |

## 1.1.0

### Inference

|   Module Name   |   Hardware Platform   |   Operating System   |   Download Links   |   SHA-256   |
|       ---       |          ---          |          ---         |         ---        |     ---     |
| MindSpore Lite Converter                          | CPU         | Ubuntu-x64                      | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/linux/mindspore-lite-1.1.0-converter-linux-x64.tar.gz> | d449e38a8493c314d1b5b1a127f62269192da785b012ff892eda775dedca3d82 |
|                                                   | CPU         | Windows-x64                     | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/windows/mindspore-lite-1.1.0-converter-win-x64.zip>      | 5e50b7701b97ebe784095f2ba954fc6c377eb157fbc9aaeae2497e38cc4ee212 |
| MindSpore Lite Runtime (include image processing) | CPU/GPU/NPU | Android-aarch64/Android-aarch32 | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/android/mindspore-lite-1.1.0-inference-android.tar.gz>   | a19de5706db57e97a5f04ef08e0e383f8ea497c70bb60e60d056b31a603c0243 |
|                                                   | CPU         | Ubuntu-x64                      | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/linux/mindspore-lite-1.1.0-inference-linux-x64.tar.gz> | 176256c2fbef775f1a44aaeccae0c4eea6a60f41fc0baece5479dcb378155f36 |
|                                                   | CPU         | Windows-x64                     | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/windows/mindspore-lite-1.1.0-inference-win-x64.zip>      | 30b5545245832a73d84732166f360c77cd09a7a4fe1fb922a8f7b80e7df326c1 |

### Train

|   Module Name   |   Hardware Platform   |   Operating System   |   Download Links   |   SHA-256   |
|       ---       |          ---          |          ---         |         ---        |     ---     |
| MindSpore Lite Converter                          | CPU         | Ubuntu-x64                      | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/linux/mindspore-lite-1.1.0-train-converter-linux-x64.tar.gz> | f95a9db98c84ec3d97f88383ecc3832582aa9737ed287c33703deb0b419acf25 |
| MindSpore Lite Runtime (include image processing) | CPU | Android-aarch64/Android-aarch32 | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/android/mindspore-lite-1.1.0-train-android.tar.gz>             | a6d8152f4e2d674c52af2c379f7d07858d30bc0dceef1dbc366e6fa16a5948b5 |
|                                                   | CPU         | Ubuntu-x64                      | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/lite/release/linux/mindspore-lite-1.1.0-train-linux-x64.tar.gz>           | 1290f0adc790adc9edce654b9a629a9a323cfcb8453eb6bc19b779ef726282bf |

> - Ubuntu-x64 Package is compiled in an environment where the GCC version is greater than or equal to 7.3.0, so the deployment environment requires the GLIBC version to be greater than or equal to 2.27.
> - Android-aarch32 does not support GPU and NPU.
> - MindSpore Lite also provides `libmindspore-lite.a` static library [cropper tool](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/cropper_tool.html#) for Runtime, which can crop the static library files, and effectively reduce the size of the library files.
> - After the download of MindSpore Lite is completed, SHA-256 integrity verification is required.
