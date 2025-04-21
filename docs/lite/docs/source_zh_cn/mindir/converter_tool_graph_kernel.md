# 图算融合配置说明（beta特性）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/converter_tool_graph_kernel.md)

## 概述

图算融合是MindSpore特有的网络性能优化技术。它可以通过自动分析和优化现有网络计算图逻辑，并结合目标硬件能力，对计算图进行计算化简和替代、算子拆分和融合、算子特例化编译等优化，以提升设备计算资源利用率，实现对网络性能的整体优化。相比传统优化技术，图算融合具有多算子跨边界联合优化、与MindSpore AKG（基于Polyhedral的算子编译器）跨层协同、即时编译等独特优势。

MindSpore Lite的whl包和tar包默认内置AKG。通过源码安装的MindSpore Lite，需确保已[安装llvm 12.0.1](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/build.html#安装llvm-可选)。通过源码安装ascend后端则需要额外安装[git-lfs](https://git-lfs.com/)。

## AKG安装方法

需要先提前安装AKG才能让MindSpore Lite使能图算功能。目前，AKG已经内置在了MindSpore Lite的发布件中，在代码编译阶段，编译MindSpore Lite的时候会同时编译AKG。AKG的安装方法有两种，这两种方式对应了MindSpore Lite的tar包形态和whl包形态的两种发布件，可以任选其一进行安装：

1. 安装tar包内置的AKG发布件

    首先，前往[下载页面](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/use/downloads.html)下载MindSpore Lite端侧推理和云侧推理的tar包发布件，然后安装tools/akg/的akg的whl包。
    接着在命令行中使用以下命令检查AKG是否安装成功：若无报错，则表示安装成功。

    ```shell
    python -c "import akg"
    ```

2. 安装MindSpore Lite的whl包

   首先，前往[下载页面](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/use/downloads.html)下载MindSpore Lite的Whl包，并使用pip进行安装。

   安装后可以使用以下命令检查MindSpore Lite内置的AKG是否安装成功：若无报错，则表示安装成功。

   ```bash
   python -c "import mindspore_lite.akg"
   ```

## 使用方法

在模型转换阶段，通过配置converter_lite 的`--configFile`和`--optimize`选项来使能图算。

首选编写配置文件：

```cfg
#文件名是akg.cfg
[graph_kernel_param]
opt_level=2
```

之后运行converter_lite进行模型转换。

编译Ascend后端的ONNX模型：

Ascend后端需要安装AKG融合算子实现，具体方法见[部署Ascend自定义算子](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool_ascend.html#%E9%83%A8%E7%BD%B2ascend%E8%87%AA%E5%AE%9A%E4%B9%89%E7%AE%97%E5%AD%90)。

部署完成之后执行如命令即可转换ONNX模型。

```bash
./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model --configFile=akg.cfg --optimize=ascend_oriented
```

之后，就可以通过benchmark工具或者模型推理接口来运行离线模型。

> 目前，使用使能图算融合之后的converter_lite工具转换出的离线模型只能在本机运行，模型文件不能支持跨平台运行功能。
>
> 可以通过导出IR图（配置`export MS_DEV_DUMP_GRAPH_KERNEL_IR=on`），查看是否生成融合后的图结构，来判断图算融合是否使能成功。
