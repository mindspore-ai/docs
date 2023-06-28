# FAQ

`Ascend` `GPU` `CPU` `Environmental Setup` `Model Export` `Model Training` `Beginner` `Intermediate` `Expert`
<a href="https://gitee.com/mindspore/docs/blob/r0.7/docs/source_en/FAQ.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Installation

### Installing Using pip
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

A: Currently, the LSTM runs only on a GPU or CPU and does not support the hardware environment. You can click [here](https://www.mindspore.cn/docs/en/r0.7/operator_list.html) to view the supported operators.

<br/>

Q: When conv2d is set to (3,10), Tensor[2,2,10,10] and it runs on Ascend on ModelArts, the error message `FM_W+pad_left+pad_right-KW>=strideW` is displayed. However, no error message is displayed when it runs on a CPU. What should I do?

A: This is a TBE operator restriction that the width of x must be greater than that of the kernel. The CPU does not have this operator restriction. Therefore, no error is reported.

## Network Models

Q: Which framework models can be directly read by MindSpore? What formats are supported?

A: MindSpore uses protocol buffers (protobuf) to store training parameters and cannot directly read framework models. If you want to use the .ckpt file trained by a framework, read the parameters and then call the save_checkpoint API of MindSpore to save the file as a .ckpt file that can be read by MindSpore.

<br/>

Q: How do I use models trained by MindSpore on Ascend 310?

A: Ascend 310 supports the offline model (OM). Therefore, you need to export the Open Neural Network Exchange (ONNX) or Ascend intermediate representation (AIR) model and then convert it into OM supported by Ascend 310. For details, see [Multi-Platform Inference](https://www.mindspore.cn/tutorial/en/r0.7/use/multi_platform_inference.html).

<br/>

Q: How do I modify parameters (such as the dropout value) on MindSpore?

A: When building a network, use `if self.training: x = dropput(x)`. During verification, set `network.set_train(mode_false)` before execution to disable the dropout function. During training, set `network.set_train(mode_false)` to True to enable the dropout function.

<br/>

Q: Where can I view the sample code or tutorial of MindSpore training and inference?

A: Please visit the [MindSpore official website](https://www.mindspore.cn/tutorial/en/r0.7/index.html).

<br/>

Q: What types of model is currently supported by MindSpore for training?

A: MindSpore has basic support for common training scenarios, please refer to [Release note](https://gitee.com/mindspore/mindspore/blob/r0.7/RELEASE.md) for detailed information.

<br/>

Q: What are the available recommendation or text generation networks or models provided by MindSpore?

A: Currently, recommendation models such as Wide & Deep, DeepFM, and NCF are under development. In the natural language processing (NLP) field, Bert\_NEZHA is available and models such as MASS are under development. You can rebuild the network into a text generation network based on the scenario requirements. Please stay tuned for updates on the [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/r0.7/model_zoo).

<br/>

Q: How simple can the MindSpore model training code be?

A: MindSpore provides Model APIs except for network definitions. In most scenarios, model training can be completed using only a few lines of code.

## Platform and System

Q: Can MindSpore be installed on Ascend 310?

A: Ascend 310 can only be used for inference. MindSpore supports training on Ascend 910. The trained model can be converted into an .om model for inference on Ascend 310.

<br/>

Q: Does MindSpore require computing units such as GPUs and NPUs? What hardware support is required?

A: MindSpore currently supports CPU, GPU, Ascend, and NPU. Currently, you can try out MindSpore through Docker images on laptops or in environments with GPUs. Some models in MindSpore Model Zoo support GPU-based training and inference, and other models are being improved. For distributed parallel training, MindSpore supports multi-GPU training. You can obtain the latest information from [Road Map](https://www.mindspore.cn/docs/en/r0.7/roadmap.html) and [project release notes](https://gitee.com/mindspore/mindspore/blob/r0.7/RELEASE.md).

<br/>

Q: Does MindSpore have any plan on supporting other types of heterogeneous computing hardwares?

A: MindSpore provides pluggable device management interface so that developer could easily integrate other types of heterogeneous computing hardwares like FPGA to MindSpore. We welcome more backend support in MindSpore from the community.

<br/>

Q: What is the relationship between MindSpore and ModelArts? Can MindSpore be used on ModelArts?

A: ModelArts is an online training and inference platform on HUAWEI CLOUD. MindSpore is a Huawei deep learning framework. You can view the tutorials on the [MindSpore official website](https://www.mindspore.cn/tutorial/zh-CN/r0.7/advanced_use/use_on_the_cloud.html) to learn how to train MindSpore models on ModelArts.

<br/>

Q: Does MindSpore support Windows 10?

A: The MindSpore CPU version can be installed on Windows 10. For details about the installation procedure, please refer to the [MindSpore official website tutorial](https://www.mindspore.cn/install/en)

## Backend Running

Q: What can I do if the error message `device target [CPU] is not supported in pynative mode` is displayed for the operation operator of MindSpore?

A: Currently, the PyNative mode supports only Ascend and GPU and does not support the CPU.

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
You can select a proper mode and writing method to complete the training by referring to the official website [tutorial](https://www.mindspore.cn/tutorial/en/r0.7/advanced_use/debugging_in_pynative_mode.html).

## Programming Language Extensions

Q: The recent announced programming language such as taichi got Python extensions that could be directly used as `import taichi as ti`. Does MindSpore have similar support?

A: MindSpore supports Python native expression via `import mindspore`。

<br/>

Q: Does MindSpore plan to support more programming languages other than Python?

A：MindSpore currently supports Python extensions，bindings for languages like C++、Rust、Julia are on the way.

## Supported Features

Q: Does MindSpore have a lightweight on-device inference engine?

A: MindSpore has its own on-device inference engine. In the current version, some functions of on-device inference have been open-sourced. MindSpore on-device inference engine is expected to be updated at the end of August. By then, it will be more comprehensive and powerful in terms of usability, performance, operator completeness, and third-party model support.

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

A: For details about script or model migration, please visit the [MindSpore official website](https://www.mindspore.cn/tutorial/en/r0.7/advanced_use/network_migration.html).

<br/>

Q: Does MindSpore provide open-source e-commerce datasets?

A: No. Please stay tuned for updates on the [MindSpore official website](https://www.mindspore.cn/en).