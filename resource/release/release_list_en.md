# Release List

<!-- TOC -->

- [Release List](#release-list)
    - [0.3.0-alpha](#030-alpha)
        - [Releasenotes](#releasenotes)
        - [Downloads](#downloads)
        - [Tutorials](#tutorials)
        - [API](#api)
        - [Docs](#docs)
    - [0.2.0-alpha](#020-alpha)
        - [Releasenotes](#releasenotes)
        - [Downloads](#downloads)
        - [Tutorials](#tutorials)
        - [API](#api)
        - [Docs](#docs)
    - [0.1.0-alpha](#010-alpha)
        - [Releasenotes](#releasenotes)
        - [Downloads](#downloads)
        - [Tutorials](#tutorials)
        - [API](#api)
        - [Docs](#docs)
    - [master(unstable)](#masterunstable)

<!-- /TOC -->

## 0.3.0-alpha

### Releasenotes

<https://gitee.com/mindspore/mindspore/blob/r0.3/RELEASE.md>

### Downloads

|   Module Name   |   Hardware Platform   |   Operating System   |      Download Links      |       SHA-256     |
|    ---   |    ---   |    ---   |       ---      |    ---      |
|   MindSpore   |   Ascend910   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindSpore/ascend/ubuntu_x86/mindspore_ascend-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   7756a50ca3af82d06eaf456db4d062fa647a8352724ef85da6569426a6393918   |
|      |      |   Ubuntu-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindSpore/ascend/ubuntu_aarch64/mindspore_ascend-0.3.0-cp37-cp37m-linux_aarch64.whl>   |   4f613b1466ba3eafb160ebca2f8086e63fdaeee9c07a5458b4476da4fce8f90a   |
|      |      |   EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindSpore/ascend/euleros_x86/mindspore_ascend-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   93867f72c801affec1da901e734a6d329c6d1ae3cdec1297870b46a277aa64b8   |
|      |      |   EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindSpore/ascend/euleros_aarch64/mindspore_ascend-0.3.0-cp37-cp37m-linux_aarch64.whl>   |   ecd7f3e049034d20f722073ecb87d5d8108cfc218d2594ec9771e83db5222cf8   |
|      |   GPU CUDA 9.2   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindSpore/gpu/ubuntu_x86/cuda-9.2/mindspore_gpu-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   cd4890d3c24b47f48da48c8cc9efdf35e14f9b4a76ec66779bb24d601d2e0c25   |
|      |   GPU CUDA 10.1   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindSpore/gpu/ubuntu_x86/cuda-10.1/mindspore_gpu-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   07e7263936e1c4805fb253d596ccbeb2fccab3a48929febce85ebb7609d82c4f   |
|      |   CPU   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindSpore/cpu/ubuntu_x86/mindspore-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   38b662673af0dfc89182f5b54261aa8694b8aefdbc1e5fa2d5e06377113e8a22   |
|      |         |   Windows-x64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindSpore/cpu/windows_x64/mindspore-0.3.0-cp37-cp37m-win_amd64.whl>   |   ed6b1c04d08fcfe4ac913f4593da70f78741af8e9391dce7189106b67a1393c1   |
|   MindInsight   |   Ascend910   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindInsight/ascend/ubuntu_x86/mindinsight-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   40b0697fbafa3a08393cbeda2f6286caa299a3b758beb63c9ed68f621879ef49   |
|      |      |   Ubuntu-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindInsight/ascend/ubuntu_aarch64/mindinsight-0.3.0-cp37-cp37m-linux_aarch64.whl>   |   0005334bf15268e499d91d0a7e1bfb5abc4b5a0e10a3c4c0798da0283b28fe23   |
|      |      |   EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindInsight/ascend/euleros_x86/mindinsight-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   e1ba11b37a0ce13c8f4f668a9479c0f97d922e4ce6128823e576c7d38298c86d   |
|      |      |   EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindInsight/ascend/euleros_aarch64/mindinsight-0.3.0-cp37-cp37m-linux_aarch64.whl>   |   8d03e1f57b39268b4ba89c25ca88934b1a00304839f454d7bfd4747269abb359   |
|      |   GPU CUDA 9.2/GPU CUDA 10.1   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindInsight/ascend/ubuntu_x86/mindinsight-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   40b0697fbafa3a08393cbeda2f6286caa299a3b758beb63c9ed68f621879ef49   |
|   MindArmour   |   Ascend910   |   Ubuntu-x86/EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindArmour/x86_64/mindarmour-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   7a2bd6174be9e5a47e8ae6bcdd592ecdafc6e53e6f1cd5f0261fcb8337b5b337   |
|      |      |   Ubuntu-aarch64/EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindArmour/aarch64/mindarmour-0.3.0-cp37-cp37m-linux_aarch64.whl>   |   6d5f96cc004579d98664d018dca860d3b7f935df5b479f1192161f18a091d9c9   |
|      |   GPU CUDA 9.2/GPU CUDA 10.1/CPU   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.3.0-alpha/MindArmour/x86_64/mindarmour-0.3.0-cp37-cp37m-linux_x86_64.whl>   |   7a2bd6174be9e5a47e8ae6bcdd592ecdafc6e53e6f1cd5f0261fcb8337b5b337   |

### Tutorials

<https://www.mindspore.cn/tutorial/en/0.3.0-alpha/index.html>

### API

<https://www.mindspore.cn/api/en/0.3.0-alpha/index.html>

### Docs

<https://www.mindspore.cn/docs/en/0.3.0-alpha/index.html>

## 0.2.0-alpha

### Releasenotes

<https://gitee.com/mindspore/mindspore/blob/r0.2/RELEASE.md>

### Downloads

|   Module Name   |   Hardware Platform   |   Operating System   |      Download Links      |       SHA-256     |
|    ---   |    ---   |    ---   |       ---      |    ---      |
|   MindSpore   |   Ascend910   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindSpore/ascend/x86_ubuntu/mindspore_ascend-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   aa1225665d05263b17bb7ec1d51dd4f933254c818bee126b6c5dac4513532a14   |
|      |      |   EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindSpore/ascend/x86_euleros/mindspore_ascend-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   eb9a1b2a0ba32d7f7264ae344833f90a8ba2042cddf1a6a719c1a38a7ea528ea   |
|      |      |   EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindSpore/ascend/aarch64_euleros/mindspore_ascend-0.2.0-cp37-cp37m-linux_aarch64.whl>   |   820fb17d63341c636018d4e930151d3d2fa7ac05d4a400286c1b1aeb4cc34c6f   |
|      |   GPU CUDA 9.2   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindSpore/gpu/cuda-9.2/mindspore_gpu-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   b933f95551afc3de38ba06502ef68a5a2a50bebadcc9b92b870f8eb44f59f10a   |
|      |   GPU CUDA 10.1   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindSpore/gpu/cuda-10.1/mindspore_gpu-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   e7167bad4549002f9d14b0a015abbabf56334621cf746fa60bb67df0fadb22ec   |
|      |   CPU   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindSpore/cpu/x86_ubuntu/mindspore-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   d6702dce9dad94d1e08bedc43540ac21422e8c49d919f7abd0bb7a3aa804476f   |
|      |         |   Windows-x64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindSpore/cpu/x64_windows/mindspore-0.2.0-cp37-cp37m-win_amd64.whl>   |   77151d20fe450df3697853a5309308ecc482870fd2984753b82d3db9d326fdec   |
|   MindInsight   |   Ascend910   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindInsight/x86_ubuntu/mindinsight-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   2334e833f322e0f38e04e65819214b7582527364c1e0aca79bd080a720932ca4   |
|      |      |   EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindInsight/x86_euleros/mindinsight-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   c6c3088a499967f2fe301ea910536fdf62dd4e38edb47e144726b9a4d4a17e50   |
|      |      |   EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindInsight/aarch64_euleros/mindinsight-0.2.0-cp37-cp37m-linux_aarch64.whl>   |   6e5e03b56988968ec36c556ece06d2e5aa68e80ff475374087998e0ff360a45a   |
|      |   GPU CUDA 9.2/GPU CUDA 10.1   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindInsight/x86_ubuntu/mindinsight-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   2334e833f322e0f38e04e65819214b7582527364c1e0aca79bd080a720932ca4   |
|   MindArmour   |   Ascend910   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindArmour/x86_64/mindarmour-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   4146790bc73a5846e92b943dfd3febb6c62052b217eeb45b6c48aa82b51e7cc3   |
|      |      |   EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindArmour/x86_64/mindarmour-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   4146790bc73a5846e92b943dfd3febb6c62052b217eeb45b6c48aa82b51e7cc3   |
|      |      |   EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindArmour/aarch64/mindarmour-0.2.0-cp37-cp37m-linux_aarch64.whl>   |   5d5e532b9c4e466d89cf503f07c2d530b42216a14f193f685b9a81e190c8db44   |
|      |   GPU CUDA 9.2/GPU CUDA 10.1/CPU   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.2.0-alpha/MindArmour/x86_64/mindarmour-0.2.0-cp37-cp37m-linux_x86_64.whl>   |   4146790bc73a5846e92b943dfd3febb6c62052b217eeb45b6c48aa82b51e7cc3   |

### Tutorials

<https://www.mindspore.cn/tutorial/en/0.2.0-alpha/index.html>

### API

<https://www.mindspore.cn/api/en/0.2.0-alpha/index.html>

### Docs

<https://www.mindspore.cn/docs/en/0.2.0-alpha/index.html>

## 0.1.0-alpha

### Releasenotes

<https://gitee.com/mindspore/mindspore/blob/r0.1/RELEASE.md>

### Downloads

|   Module Name   |   Hardware Platform   |   Operating System   |      Download Links      |       SHA-256     |
|    ---   |    ---   |    ---   |       ---      |    ---      |
|   MindSpore   |   Ascend910   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindSpore/ascend/ubuntu-x86/mindspore-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   a76df4e96c4cb69b10580fcde2d4ef46b5d426be6d47a3d8fd379c97c3e66638   |
|      |      |   EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindSpore/ascend/euleros-x86/mindspore-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   45d4fcb37bf796b3208b7c1ca70dc0db1387a878ef27836d3d445f311c8c02e0   |
|      |      |   EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindSpore/ascend/euleros-aarch64/mindspore-0.1.0-cp37-cp37m-linux_aarch64.whl>   |   7daba2d1739ce19d55695460dce5ef044b4d38baad4f5117056e5f77f49a12b4   |
|      |   GPU CUDA 9.2   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindSpore/gpu/cuda-9.2/mindspore-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   b6e5623135b57b8c262f3e32d97fbe1e20e8c19da185a7aba97b9dc98c7ecda1   |
|      |   GPU CUDA 10.1   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindSpore/gpu/cuda-10.1/mindspore-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   43711725cf7e071ca21b5ba25e90d6955789fe3495c62217e70869f52ae20c01   |
|      |   CPU   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindSpore/cpu/ubuntu-x86/mindspore-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   45c473a97a6cb227e4221117bfb1b3ebe3f2eab938e0b76d5117e6c3127b8e5c   |
|   MindInsight   |   Ascend910   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindInsight/ubuntu/x86_64/mindinsight-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   960b6f485ce545ccce98adfb4c62cdea216c9b7851ffdc0669827c53811c3e59   |
|      |      |   EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindInsight/euleros/x86_64/mindinsight-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   9f1ef04fec09e5b90be4a6223b3bf2943334746c1f5dac37207db4524b64942f   |
|      |      |   EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindInsight/euleros/aarch64/mindinsight-0.1.0-cp37-cp37m-linux_aarch64.whl>   |   d64207126542571057572f856010a5a8b3362ccd9e5b5c81da5b78b94face5fe   |
|   MindArmour   |   Ascend910   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindArmour/x86_64/mindarmour-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   7796b6c114ee4962ce605da59a9bc47390c8910acbac318ecc0598829aad6e8c   |
|      |      |   EulerOS-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindArmour/x86_64/mindarmour-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   7796b6c114ee4962ce605da59a9bc47390c8910acbac318ecc0598829aad6e8c   |
|      |      |   EulerOS-aarch64   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindArmour/aarch64/mindarmour-0.1.0-cp37-cp37m-linux_aarch64.whl>   |   f354fcdbb3d8b4022fda5a6636e763f8091aca2167dc23e60b7f7b6d710523cb   |
|      |   GPU CUDA 9.2/GPU CUDA 10.1/CPU   |   Ubuntu-x86   |   <https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindArmour/x86_64/mindarmour-0.1.0-cp37-cp37m-linux_x86_64.whl>   |   7796b6c114ee4962ce605da59a9bc47390c8910acbac318ecc0598829aad6e8c   |

### Tutorials

<https://www.mindspore.cn/tutorial/en/0.1.0-alpha/index.html>

### API

<https://www.mindspore.cn/api/en/0.1.0-alpha/index.html>

### Docs

<https://www.mindspore.cn/docs/en/0.1.0-alpha/index.html>

## master(unstable)

### Tutorials

<https://www.mindspore.cn/tutorial/en/master/index.html>

### API

<https://www.mindspore.cn/api/en/master/index.html>

### Docs

<https://www.mindspore.cn/docs/en/master/index.html>
