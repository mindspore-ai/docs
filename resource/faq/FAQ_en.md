# FAQ

## Installation

### Pip Installation

Q: Any specific requirements for Python version when pip install MindSpore?

A: MindSpore utilizes many of the new features in Python3.7+，therefore we recommend you add Python3.7.5 develop environment via `conda`.

<br/>

Q: What should I do when error prompts during pip install ?

A: Please execute `pip -V` to check if pip is linked to Python3.7+. If not, we recommend you
use `python3.7 -m pip install` instead of `pip install` command.

<br/>

Q: What should I do if I cannot find whl package for MindInsight or MindArmour for installation ?

A: You can download whl package from the official [MindSpore Website download page](https://www.mindspore.cn/versions) and manually install it via `pip install`.

### Source Code Installation

Q: What should I do if the compilation time of MindSpore source code taking too long or the process is constantly interrupted by errors ?

A: MindSpore imports third party dependencies through submodule mechanism, among which `protobuf` v3.8.0 might not have the optimal or steady download speed, we recommend you prepare the protobuf package beforehand via other method.

<br/>

Q: How to change installation directory of the third party libraries?

A: The third party libraries will be installed in build/mindspore/.mslib, you can change the installation directory by setting the environment variable MSLIBS_CACHE_PATH, eg. `export MSLIBS_CACHE_PATH = ~/.mslib`.

<br/>

Q: What should I do if the software version required by MindSpore is not the same with the Ubuntu default software version ?

A: At the moment some software might need manual upgrade. (**Note**：MindSpore requires Python3.7.5 and gcc7.3，the default version in Ubuntu 16.04 are Python3.5 and gcc5，whereas the one in Ubuntu 18.04 are Python3.7.3 and gcc7.4)

<br/>

Q: What should I do if there is a prompt `tclsh not found` when I compile MindSpore from source code ?

A: Please install the software manually if there is any suggestion of certain `software not found`.

## Support

### Model Support

Q: What types of model is currently supported by MindSpore for training ?

A: MindSpore has basic support for common training scenarios, please refer to [Release note](https://gitee.com/mindspore/mindspore/blob/r0.5/RELEASE.md) for detailed information.

<br/>

Q: What are the available recommendation or text generation networks or models provided by MindSpore?

A: Currently, recommendation models such as Wide & Deep, DeepFM, and NCF are under development. In the natural language processing (NLP) field, Bert\_NEZHA is available and models such as MASS are under development. You can rebuild the network into a text generation network based on the scenario requirements. Please stay tuned for updates on the [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo).

### Backend Support

Q: When install or run MindSpore, are there any requirements for hardwares like GPU, NPU and so forth ?

A: MindSpore currently supports Ascend AI processor, CPU and GPU。For common models like lenet you can try run MindSpore on CPU alone.

<br/>

Q: Does MindSpore have any plan on supporting other types of heterogeneous computing hardwares ?

A: MindSpore provides pluggable device management interface so that developer could easily integrate other types of heterogeneous computing hardwares like FPGA to MindSpore. We welcome more backend support in MindSpore from the community.

<br/>

Q: What hardware does MindSpore require?

A: Currently, you can try out MindSpore through Docker images on laptops or in environments with GPUs. Some models in MindSpore Model Zoo support GPU-based training and inference, and other models are being improved. For distributed parallel training, MindSpore supports multi-GPU training. You can obtain the latest information from [RoadMap](https://www.mindspore.cn/docs/en/r0.5/roadmap.html) and project [Release Notes](https://gitee.com/mindspore/mindspore/blob/r0.5/RELEASE.md).

### System Support

Q: Does MindSpore support Windows 10?

A: The MindSpore CPU version can be installed on Windows 10.

### Programming Language

Q: The recent announced programming language such as taichi got Python extensions that could be directly used as `import taichi as ti`. Does MindSpore have similar support ?

A: MindSpore supports Python native expression via `import mindspore`。

<br/>

Q: Does MindSpore plan to support more programming languages other than Python ?

A：MindSpore currently supports Python extensions，bindings for languages like C++、Rust、Julia are on the way.

### Others

Q: How does MindSpore implement semantic collaboration and processing? Is the popular Formal Concept Analysis (FCA) used?

A: The MindSpore framework does not support FCA. For semantic models, you can call third-party tools to perform FCA in the data preprocessing phase. MindSpore supports Python therefore `import FCA` could do the trick.

<br/>

Q: Where can I view the sample code or tutorial of MindSpore training and inference?

A: Please visit the [MindSpore official website](https://www.mindspore.cn/tutorial/en/r0.5/index.html).

## Features

Q: Does MindSpore have any plan or consideration on the edge and device when the training and inference functions on the cloud are relatively mature?

A: MindSpore is a unified cloud-edge-device training and inference framework. Edge has been considered in its design, so MindSpore can perform inference at the edge. The open-source version will support Ascend 310-based inference. Currently, inference supports optimization operations, including quantization, operator fusion, and memory overcommitment.

<br/>

Q: How does MindSpore support automatic parallelism?

A: Automatic parallelism on CPUs and GPUs are being improved. You are advised to use the automatic parallelism feature on the Ascend 910 AI processor. Follow our open source community and apply for a MindSpore developer experience environment for trial use.

<br/>

Q: What is the relationship between MindSpore and ModelArts? Can MindSpore be used on ModelArts?

A: ModelArts is an online training and inference platform on HUAWEI CLOUD. MindSpore is a Huawei deep learning framework. You can view the tutorials on the [MindSpore official website](https://www.mindspore.cn/tutorial/zh-CN/r0.5/advanced_use/use_on_the_cloud.html) to learn how to train MindSpore models on ModelArts.

## Capabilities

Q: Does MindSpore have a module that can implement object detection algorithms as TensorFlow does?

A: The TensorFlow's object detection pipeline API belongs to the TensorFlow's Model module. After MindSpore's detection models are complete, similar pipeline APIs will be provided.

<br/>

Q: How do I migrate scripts or models of other frameworks to MindSpore?

A: For details about script or model migration, please visit the [MindSpore official website](https://www.mindspore.cn/tutorial/en/r0.5/advanced_use/network_migration.html).

<br/>

Q: Does MindSpore provide open-source e-commerce datasets?

A: No. Please stay tuned for updates on the [MindSpore official website](https://www.mindspore.cn/en).

<br/>

Q: How simple can the MindSpore model training code be?

A: MindSpore provides Model APIs except for network definitions. In most scenarios, model training can be completed using only a few lines of code.
