# Ascend 310 AI处理器上使用MindIR模型进行推理

`Linux` `Ascend` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- [Ascend 310 AI处理器上使用MindIR模型进行推理](#ascend-310-ai处理器上使用mindir模型进行推理)
    - [概述](#概述)
    - [开发环境准备](#开发环境准备)
    - [推理目录结构介绍](#推理目录结构介绍)
    - [导出MindIR模型文件](#导出mindir模型文件)
    - [编译推理代码](#编译推理代码)
    - [执行推理并查看结果](#执行推理并查看结果)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/inference/source_zh_cn/multi_platform_inference_ascend_310_mindir.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 概述

Ascend 310是面向边缘场景的高能效高集成度AI处理器。Atlas 200开发者套件又称Atlas 200 Developer Kit（以下简称Atlas 200 DK），是以Atlas 200 AI加速模块为核心的开发者板形态的终端类产品，集成了海思Ascend 310 AI处理器，可以实现图像、视频等多种数据分析与推理计算，可广泛用于智能监控、机器人、无人机、视频服务器等场景。

本教程介绍如何在Atlas 200 DK上使用MindSpore执行推理，主要包括以下流程：

1. 开发环境准备，包括制作Atlas 200 DK的SD卡 、配置Python环境和刷配套开发软件包。

2. 导出MindIR模型文件，这里以ResNet-50模型为例。

3. 编译推理代码，生成可执行`main`文件。

4. 加载保存的MindIR模型，执行推理并查看结果。

> 你可以在这里找到完整可运行的样例代码：<https://gitee.com/mindspore/docs/tree/r1.1/tutorials/tutorial_code/ascend310_resnet50_preprocess_sample> 。

## 开发环境准备

参考[Ascend 310 AI处理器上使用AIR进行推理#开发环境准备](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.1/multi_platform_inference_ascend_310_air.html#id2)

## 推理目录结构介绍

创建目录放置推理代码工程，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample`，目录代码可以从官网示例下载[样例代码](https://gitee.com/mindspore/docs/tree/r1.1/tutorials/tutorial_code/ascend310_resnet50_preprocess_sample)，`model`目录用于存放接下来导出的`MindIR`模型文件，`test_data`目录用于存放待分类的图片，推理代码工程目录结构如下:

```text
└─ascend310_resnet50_preprocess_sample
    ├── CMakeLists.txt                    // 编译脚本
    ├── README.md                         // 使用说明
    ├── main.cc                           // 主函数
    ├── model
    │   └── resnet50_imagenet.mindir      // MindIR模型文件
    └── test_data
        ├── ILSVRC2012_val_00002138.JPEG  // 输入样本图片1
        ├── ILSVRC2012_val_00003014.JPEG  // 输入样本图片2
        ├── ...                           // 输入样本图片n
```

## 导出MindIR模型文件

在Ascend 910的机器上训练好目标网络，并保存为CheckPoint文件，通过网络和CheckPoint文件导出对应的MindIR格式模型文件，导出流程参见[导出MindIR格式文件](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/use/save_model.html#mindir)。

> 这里提供使用ResNet-50模型导出的示例MindIR文件[resnet50_imagenet.mindir](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir)。

## 编译推理代码

进入工程目录`ascend310_resnet50_preprocess_sample`，设置如下环境变量：

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/acllib/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/atc/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/atc/ccec_compiler/bin/:${PATH}                       # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

执行`cmake`命令：

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
```

再执行`make`命令编译即可。

```bash
make
```

编译完成后，在`ascend310_resnet50_preprocess_sample`下会生成可执行`main`文件。

## 执行推理并查看结果

登录Atlas 200 DK开发者板环境，创建`model`目录放置MindIR文件`resnet50_imagenet.mindir`，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample/model`。
创建`test_data`目录放置图片，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample/test_data`。
就可以开始执行推理了:

```bash
./resnet50_sample
```

执行后，会对`test_data`目录下放置的所有图片进行推理，比如放置了9张[ImageNet2012](http://image-net.org/download-images)验证集中label为0的图片，可以看到推理结果如下。

```text
Image: ./test_data/ILSVRC2012_val_00002138.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00003014.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00006697.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00007197.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009111.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009191.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009346.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009379.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009396.JPEG infer result: 0
```
