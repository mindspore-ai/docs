# Graph Kernel Fusion Configuration Instructions (Beta Feature)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/mindir/converter_tool_graph_kernel.md)

## Introduction

Graph kernel fusion is a unique network performance optimization technique in MindSpore. It can automatically analyze and optimize the existing network computational graph logic and combine with the target hardware capabilities to perform optimizations, such as computational simplification and substitution, operator splitting and fusion, operator special case compilation, to improve the utilization of device computational resources and achieve the overall optimization of network performance. Compared with traditional optimization techniques, graph kernel fusion has unique advantages such as joint optimization of multiple operators across boundaries, cross-layer collaboration with MindSpore AKG (Polyhedral-based operator compiler), and on-the-fly compilation.

MindSpore Lite whl and tar packages have built-in AKG by default. For the installed MindSpore Lite via source code, make sure you have [installed llvm 12.0.1](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/build.html#installing-llvm-optional). Installing the ascend backend via source code requires an additional installation of [git-lfs](https://git-lfs.com/).

## AKG Installation

Install AKG in advance to enable graph kernel in the MindSpore Lite. Currently, AKG is built into the MindSpore Lite distribution. During the code compilation phase, AKG is compiled at the same time when MindSpore Lite is compiled. There are two ways to install AKG, which correspond to the two distributions of MindSpore Lite in tar and whl packages. You can choose either one to install:

1. Install the AKG distribution built in the tar package

    First, go to [download page](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) to download the tar distribution for MindSpore Lite device-side inference and cloud-side inference, and then install the akg package in tools/akg/. Next, use the following command in the command line to check whether AKG is installed successfully: if no error is reported, the installation is successful.

    ```shell
    python -c "import akg"
    ```

2. Install the whl package of MindSpore Lite

    First, go to [download page](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) to download the Whl package of MindSpore Lite and install it using pip.

    After installation, you can use the following command to check if built-in AKG of MindSpore Lite is successfully installed: If no error is reported, the installation is successful.

    ```bash
   python -c "import mindspore_lite.akg"
   ```

## Usage

During the model conversion phase, the graph kernel is enabled by configuring the `--configFile` and `--optimize` options of converter_lite.

Writing configuration file is preferred:

```cfg
# The file name is akg.cfg
[graph_kernel_param]
opt_level=2
```

After that, run converter_lite to perform the model conversion.

Compiling the ONNX model on the Ascend backend:

The Ascend backend requires the installation of the AKG fusion operator, as described in [Deploying Ascend Custom Operators](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool_ascend.html#deploying-ascend-custom-operators).

After deployment, execute the following command to convert the ONNX model.

```bash
./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model --configFile=akg.cfg --optimize=ascend_oriented
```

After that, the offline model can be run through the benchmark tool or the model inference interface.

> At present, the offline models converted by the converter_lite tool after graph kernel fusion is enabled can only be run locally, and the model files cannot support cross-platform running function.
>
> You can determine whether graph fusion is successful by exporting the IR graph (configuring `export MS_DEV_DUMP_GRAPH_KERNEL_IR=on`) and checking if the fused graph structure has been generated.
