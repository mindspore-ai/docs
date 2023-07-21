# Overview

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.2/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/tutorials/source_en/introduction.md)

The following describes the Huawei AI full-stack solution and introduces the position of MindSpore in the solution. Developers who are interested in MindSpore can visit the [MindSpore community](https://gitee.com/mindspore/mindspore) and click [Watch, Star, and Fork](https://gitee.com/mindspore/mindspore).

## Introduction to Huawei Ascend AI Full-Stack Solution

Ascend computing is a full-stack AI computing infrastructure and application based on the Ascend series processors. It includes the Ascend series chips, Atlas series hardware, CANN chip enablement, MindSpore AI framework, ModelArts, and MindX application enablement.

Huawei Atlas AI computing solution is based on Ascend series AI processors and uses various product forms such as modules, cards, edge stations, servers, and clusters to build an all-scenario AI infrastructure solution oriented to device, edge, and cloud. It covers data center and intelligent edge solutions, as well as the entire inference and training processes in the deep learning field.

- **Atlas series**: provides AI training, inference cards, and training servers ([learn more](https://e.huawei.com/en/products/cloud-computing-dc/atlas/)).
- **CANN at heterogeneous computing architecture**: a driver layer that enables chips ([learn more](https://www.hiascend.com/en/software/cann)).
- **MindSpore**: all-scenario AI framework ([learn more](https://www.mindspore.cn/en)).
- **MindX SDK**: Ascend SDK that provides application solutions ([learn more](https://www.hiascend.com/en/software/mindx-sdk)).
- **ModelArts**: HUAWEI CLOUD AI development platform ([learn more](https://www.huaweicloud.com/product/modelarts.html)).
- **MindStudio**: E2E development toolchain that provides one-stop IDE for AI development ([learn more](https://www.hiascend.com/en/software/mindstudio)).

For details, click [Huawei Ascend official website](https://e.huawei.com/en/products/servers/ascend).

## MindSpore Introduction

MindSpore is a deep learning framework in all scenarios, aiming to achieve easy development, efficient execution, and all-scenario coverage. Easy development features friendly APIs and easy debugging. Efficient execution is reflected in computing, data preprocessing, and distributed training. All-scenario coverage means that the framework supports cloud, edge, and device scenarios.

The following figure shows the overall MindSpore architecture, which mainly consists of four parts: MindSpore Extend, MindExpress (ME), MindCompiler, and MindRE.

- **MindSpore Extend**: MindSpore extension package to be contributed and built by more developers.
- **MindExpress**: Python-based frontend expression. In the future, more frontends based on C/C++ and Java will be provided. Cangjie, Huawei's self-developed programming language frontend, is now in the pre-research phase. In addition, Huawei is working on interconnection with third-party frontends such as Julia to introduce more third-party ecosystems.
- **MindCompiler**: core compiler of the layer, which implements three major functions based on the unified device-cloud MindIR, including hardware-independent optimization (type derivation, automatic differentiation, and expression simplification), hardware-related optimization (automatic parallelism, memory optimization, graph kernel fusion, and pipeline execution) and optimization related to deployment and inference (quantification and pruning). MindAKG is the automatic operator generation compiler of MindSpore and is being improved.
- **MindRE**: all-scenario runtime, which covers the cloud, device, and smaller IoT scenarios.

![MindSpore](images/introduction2.png)

### API Level Structure

To support network building, entire graph execution, subgraph execution, and single-operator execution, MindSpore provides users with three levels of APIs. In ascending order, these are Low-Level Python API, Medium-Level Python API, and High-Level Python API.

![MindSpore API](images/introduction3.png)

- High-Level Python API

  High-level APIs are at the first layer. Based on the medium-level API, these advanced APIs include training and inference management, mixed precision training, and debugging and optimization, enabling users to control the execution process of the entire network and implement training, inference, and optimization of the neural network. For example, by utilizing the Model API, users can specify the neural network model to be trained as well as related training settings, train the neural network model, and debug the neural network performance through the Profiler API.

- Medium-Level Python API

  Medium-level APIs are at the second layer, which encapsulates low-cost APIs and provides such modules as the network layer, optimizer, and loss function. Users can flexibly build neural networks and control execution processes through the medium-level API to quickly implement model algorithm logic. For example, users can call the Cell API to build neural network models and computing logic, add the loss function and optimization methods to the neural network model by using the loss module and Optimizer API, and use the dataset module to process data for model training and derivation.

- Low-Level Python API

  Low-level APIs are at the third layer, including tensor definition, basic operators, and automatic differential modules, enabling users to easily define tensors and perform derivative computation. For example, users can customize tensors by using the Tensor API, and use the GradOperation operator in the ops.composite module to calculate the derivative of the function at a specified position.

## Joining the Community

Welcome every developer to the MindSpore community and contribute to this all-scenario AI framework.

- **MindSpore official website**: provides comprehensive MindSpore information, including installation, tutorials, documents, community, resources, and news ([learn more](https://www.mindspore.cn/en)).
- **MindSpore code**:

    - [MindSpore Gitee](https://gitee.com/mindspore/mindspore): Top 1 Gitee open-source project in 2020, where you can track the latest progress of MindSpore by clicking Watch, Star, and Fork, discuss issues, and commit code.

    - [MindSpore Github](https://github.com/mindspore-ai/mindspore): MindSpore code image of Gitee. Developers who are accustomed to using GitHub can learn MindSpore and view the latest code implementation here.

- **MindSpore forum**: We are dedicated to serving every developer. You can find your voice in MindSpore, regardless of whether you are an entry-level developer or a master. Let's learn and grow together. ([Learn more](https://bbs.huaweicloud.com/forum/forum-1076-1.html))
