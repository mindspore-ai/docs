# FAQ

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environmental Setup` `Model Export` `Model Training` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [FAQ](#faq)
    - [Installation](#installation)
        - [Installing Using pip](#installing-using-pip)
        - [Source Code Compilation Installation](#source-code-compilation-installation)
        - [Environment Variables](#environment-variables)
        - [Verifying the Installation](#verifying-the-installation)
    - [Supported Operators](#supported-operators)
    - [Network Models](#network-models)
    - [Platform and System](#platform-and-system)
    - [Backend Running](#backend-running)
    - [Programming Language Extensions](#programming-language-extensions)
    - [Supported Features](#supported-features)

<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/faq/source_en/faq.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Installation

### Installing Using pip

Q: What can I do if an error message `cannot open shared object file:file such file or directory` is displayed when I install MindSpore of the GPU, CUDA 10.1, 0.5.0-beta, or Ubuntu-x86 version?

A: The error message indicates that the cuBLAS library is not found. Generally, the cause is that the cuBLAS library is not installed or is not added to the environment variable. Generally, cuBLAS is installed together with CUDA and the driver. After the installation, add the directory where cuBLAS is located to the `LD_LIBRARY_PATH` environment variable.

<br/>

Q: What should I do if an error message `SSL:CERTIFICATE_VERIFY_FATLED` is displayed when I use pip to install MindSpore?

A: Add the `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com` parameter to the pip installation command and try again.

<br/>

Q: Any specific requirements for Python version when pip install MindSpore?

A: MindSpore utilizes many of the new features in Python3.7+，therefore we recommend you add Python3.7.5 develop environment via `conda`.

<br/>

Q：What should I do when error `ProxyError(Cannot connect to proxy)` prompts during pip install?

A：It is generally a proxy configuration problem, you can using `export http_proxy={your_proxy}` on Ubuntu environment, and using `set http_proxy={your_proxy}` in cmd on Windows environment to config your proxy.

<br/>

Q: What should I do when error prompts during pip install?

A: Please execute `pip -V` to check if pip is linked to Python3.7+. If not, we recommend you
use `python3.7 -m pip install` instead of `pip install` command.

<br/>

Q: What should I do if I cannot find whl package for MindInsight or MindArmour on the installation page of MindSpore website?

A: You can download whl package from the official [MindSpore Website download page](https://www.mindspore.cn/versions) and manually install it via `pip install`.

### Source Code Compilation Installation

Q: A sample fails to be executed after I installed MindSpore 0.6.0 beta on Ascend 910 using Ubuntu_aarch64 and Python 3.7.5 and manually downloaded the .whl package of the corresponding version, compiled and installed GMP6.1.2, and installed other Python library dependencies. An error message is displayed, indicating that the .so file cannot be found. What can I do?

A: The `libdatatransfer.so` dynamic library is in the `fwkacllib/lib64` directory. Find the path of the library in the `/usr/local` directory, and then add the path to the `LD_LIBRARY_PATH` environment variable. After the settings take effect, execute the sample again.

<br/>

Q: What should I do if the compilation time of MindSpore source code takes too long or the process is constantly interrupted by errors?

A: MindSpore imports third party dependencies through submodule mechanism, among which `protobuf` v3.8.0 might not have the optimal or steady download speed, it is recommended that you perform package cache in advance.

<br/>

Q: How to change installation directory of the third party libraries?

A: The third party libraries will be installed in build/mindspore/.mslib, you can change the installation directory by setting the environment variable MSLIBS_CACHE_PATH, eg. `export MSLIBS_CACHE_PATH = ~/.mslib`.

<br/>

Q: What should I do if the software version required by MindSpore is not the same with the Ubuntu default software version?

A: At the moment some software might need manual upgrade. (**Note**：MindSpore requires Python3.7.5 and gcc7.3，the default version in Ubuntu 16.04 are Python3.5 and gcc5，whereas the one in Ubuntu 18.04 are Python3.7.3 and gcc7.4)

<br/>

Q: What should I do if there is a prompt `tclsh not found` when I compile MindSpore from source code?

A: Please install the software manually if there is any suggestion of certain `software not found`.

### Environment Variables

Q：Some frequently-used environment settings need to be reset in the newly started terminal window, which is easy to be forgotten, What should I do?

A：You can write the frequently-used environment settings to `~/.bash_profile` or `~/.bashrc` so that the settings can take effect immediately when you start a new terminal window.

### Verifying the Installation

Q: After MindSpore is installed on a CPU of a PC, an error message `the pointer[session] is null` is displayed during code verification. The specific code is as follows. How do I verify whether MindSpore is successfully installed?
```python
import numpy as np
from mindspore import Tensor
from mindspore.ops import functional as F
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(F.tensor_add(x,y))
```

A: After MindSpore is installed on a CPU hardware platform, run the `python -c'import mindspore'` command to check whether MindSpore is successfully installed. If no error message such as `No module named'mindspore'` is displayed, MindSpore is successfully installed. The verification code is used only to verify whether a Ascend platform is successfully installed.

## Supported Operators

Q: What can I do if the LSTM example on the official website cannot run on Ascend?

A: Currently, the LSTM runs only on a GPU or CPU and does not support the hardware environment. You can click [here](https://www.mindspore.cn/doc/note/en/r1.0/operator_list_ms.html) to view the supported operators.

<br/>

Q: When conv2d is set to (3,10), Tensor[2,2,10,10] and it runs on Ascend on ModelArts, the error message `FM_W+pad_left+pad_right-KW>=strideW` is displayed. However, no error message is displayed when it runs on a CPU. What should I do?

A: This is a TBE operator restriction that the width of x must be greater than that of the kernel. The CPU does not have this operator restriction. Therefore, no error is reported.

## Network Models

Q: What framework models and formats can be directly read by MindSpore? Can the PTH Model Obtained Through Training in PyTorch Be Loaded to the MindSpore Framework for Use?

A: MindSpore uses protocol buffers (protobuf) to store training parameters and cannot directly read framework models. A model file stores parameters and their values. You can use APIs of other frameworks to read parameters, obtain the key-value pairs of parameters, and load the key-value pairs to MindSpore. If you want to use the .ckpt file trained by a framework, read the parameters and then call the `save_checkpoint` API of MindSpore to save the file as a .ckpt file that can be read by MindSpore.

<br/>

Q: How do I use models trained by MindSpore on Ascend 310? Can they be converted to models used by HiLens Kit?

A: Yes. HiLens Kit uses Ascend 310 as the inference core. Therefore, the two questions are essentially the same. Ascend 310 requires a dedicated OM model. Use MindSpore to export the ONNX or AIR model and convert it into an OM model supported by Ascend 310. For details, see [Multi-platform Inference](https://www.mindspore.cn/tutorial/inference/en/r1.0/multi_platform_inference_ascend_310.html).

<br/>

Q: How do I modify parameters (such as the dropout value) on MindSpore?

A: When building a network, use `if self.training: x = dropput(x)`. During verification, set `network.set_train(mode_false)` before execution to disable the dropout function. During training, set `network.set_train(mode_false)` to True to enable the dropout function.

<br/>

Q: Where can I view the sample code or tutorial of MindSpore training and inference?

A: Please visit the [MindSpore official website training](https://www.mindspore.cn/tutorial/training/en/r1.0/index.html) and [MindSpore official website inference](https://www.mindspore.cn/tutorial/inference/en/r1.0/index.html).

<br/>

Q: What types of model is currently supported by MindSpore for training?

A: MindSpore has basic support for common training scenarios, please refer to [Release note](https://gitee.com/mindspore/mindspore/blob/r1.0/RELEASE.md) for detailed information.

<br/>

Q: What are the available recommendation or text generation networks or models provided by MindSpore?

A: Currently, recommendation models such as Wide & Deep, DeepFM, and NCF are under development. In the natural language processing (NLP) field, Bert\_NEZHA is available and models such as MASS are under development. You can rebuild the network into a text generation network based on the scenario requirements. Please stay tuned for updates on the [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/r1.0/model_zoo).

<br/>

Q: How simple can the MindSpore model training code be?

A: MindSpore provides Model APIs except for network definitions. In most scenarios, model training can be completed using only a few lines of code.

## Platform and System

Q: Can MindSpore be installed on Ascend 310?

A: Ascend 310 can only be used for inference. MindSpore supports training on Ascend 910. The trained model can be converted into an .om model for inference on Ascend 310.

<br/>

Q: Does MindSpore require computing units such as GPUs and NPUs? What hardware support is required?

A: MindSpore currently supports CPU, GPU, Ascend, and NPU. Currently, you can try out MindSpore through Docker images on laptops or in environments with GPUs. Some models in MindSpore Model Zoo support GPU-based training and inference, and other models are being improved. For distributed parallel training, MindSpore supports multi-GPU training. You can obtain the latest information from [Road Map](https://www.mindspore.cn/doc/note/en/r1.0/roadmap.html) and [project release notes](https://gitee.com/mindspore/mindspore/blob/r1.0/RELEASE.md).

<br/>

Q: Does MindSpore have any plan on supporting other types of heterogeneous computing hardwares?

A: MindSpore provides pluggable device management interface so that developer could easily integrate other types of heterogeneous computing hardwares like FPGA to MindSpore. We welcome more backend support in MindSpore from the community.

<br/>

Q: Does MindSpore support Windows 10?

A: The MindSpore CPU version can be installed on Windows 10. For details about the installation procedure, please refer to the [MindSpore official website tutorial](https://www.mindspore.cn/install/en)

## Backend Running

Q: What can I do if an error message `wrong shape of image` is displayed when I use a model trained by MindSpore to perform prediction on a `28 x 28` digital image with white text on a black background?

A: The MNIST gray scale image dataset is used for MindSpore training. Therefore, when the model is used, the data must be set to a `28 x 28 `gray scale image, that is, a single channel.

<br/>

Q: What can I do if the error message `device target [CPU] is not supported in pynative mode` is displayed for the operation operator of MindSpore?

A: Currently, the PyNative mode supports only Ascend and GPU and does not support the CPU.

<br/>

Q: For Ascend users, how to get more detailed logs when the `run time error` is reported?

A: More detailed logs info can be obtained by modify slog config file. You can get different level by modify `/var/log/npu/conf/slog/slog.conf`. The values are as follows: 0:debug、1:info、2:warning、3:error、4:null(no output log), default 1.

<br/>

Q: What can I do if the error message `Pynative run op ExpandDims failed` is displayed when the ExpandDims operator is used? The code is as follows:

```python
context.set_context(
mode=cintext.GRAPH_MODE,
device_target='ascend')
input_tensor=Tensor(np.array([[2,2],[2,2]]),mindspore.float32)
expand_dims=P.ExpandDims()
output=expand_dims(input_tensor,0)
```

A: The problem is that the Graph mode is selected but the PyNative mode is used. As a result, an error is reported. MindSpore supports the following running modes which are optimized in terms of debugging or running:

- PyNative mode: dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.

- Graph mode: static graph mode. In this mode, the neural network model is compiled into an entire graph and then delivered for execution. This mode uses technologies such as graph optimization to improve the running performance and facilitates large-scale deployment and cross-platform running.
You can select a proper mode and writing method to complete the training by referring to the official website [tutorial](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/debug_in_pynative_mode.html).

## Programming Language Extensions

Q: The recent announced programming language such as taichi got Python extensions that could be directly used as `import taichi as ti`. Does MindSpore have similar support?

A: MindSpore supports Python native expression via `import mindspore`。

<br/>

Q: Does MindSpore plan to support more programming languages other than Python?

A：MindSpore currently supports Python extensions，bindings for languages like C++、Rust、Julia are on the way.

## Supported Features

Q: What are the advantages and features of MindSpore parallel model training?

A: In addition to data parallelism, MindSpore distributed training also supports operator-level model parallelism. The operator input tensor can be tiled and parallelized. On this basis, automatic parallelism is supported. You only need to write a single-device script to automatically tile the script to multiple nodes for parallel execution.

<br/>

Q: Has MindSpore implemented the anti-pooling operation similar to `nn.MaxUnpool2d`?
A: Currently, MindSpore does not provide anti-pooling APIs but you can customize the operator to implement the operation. For details, click [here](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/custom_operator_ascend.html).

<br/>

Q: Does MindSpore have a lightweight on-device inference engine?

A:The MindSpore lightweight inference framework MindSpore Lite has been officially launched in r0.7. You are welcome to try it and give your comments. For details about the overview, tutorials, and documents, see [MindSpore Lite](https://www.mindspore.cn/lite/en).

<br/>

Q: How does MindSpore implement semantic collaboration and processing? Is the popular Formal Concept Analysis (FCA) used?

A: The MindSpore framework does not support FCA. For semantic models, you can call third-party tools to perform FCA in the data preprocessing phase. MindSpore supports Python therefore `import FCA` could do the trick.

<br/>

Q: Does MindSpore have any plan or consideration on the edge and device when the training and inference functions on the cloud are relatively mature?

A: MindSpore is a unified cloud-edge-device training and inference framework. Edge has been considered in its design, so MindSpore can perform inference at the edge. The open-source version will support Ascend 310-based inference. The optimizations supported in the current inference stage include quantization, operator fusion, and memory overcommitment.

<br/>

Q: How does MindSpore support automatic parallelism?

A: Automatic parallelism on CPUs and GPUs are being improved. You are advised to use the automatic parallelism feature on the Ascend 910 AI processor. Follow our open source community and apply for a MindSpore developer experience environment for trial use.

<br/>

Q: Does MindSpore have a module that can implement object detection algorithms as TensorFlow does?

A: The TensorFlow's object detection pipeline API belongs to the TensorFlow's Model module. After MindSpore's detection models are complete, similar pipeline APIs will be provided.

<br/>

Q: How do I migrate scripts or models of other frameworks to MindSpore?

A: For details about script or model migration, please visit the [MindSpore official website](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/migrate_3rd_scripts.html).

<br/>

Q: Does MindSpore provide open-source e-commerce datasets?

A: No. Please stay tuned for updates on the [MindSpore official website](https://www.mindspore.cn/en).