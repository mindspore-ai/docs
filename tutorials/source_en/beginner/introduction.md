[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/introduction.md)

**Introduction** || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html#) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/master/beginner/accelerate_with_static_graph.html)

# Overview

The following describes the Huawei AI full-stack solution and the position of MindSpore in the solution. Developers who are interested in MindSpore can visit the [MindSpore community](https://gitee.com/mindspore/mindspore) and click [Watch, Star, and Fork](https://gitee.com/mindspore/mindspore).

## Introduction to MindSpore

MindSpore is a deep learning framework in all scenarios, aiming to achieve easy development, efficient execution, and unified deployment for all scenarios.

Easy development features user-friendly APIs and low debugging difficulty. Efficient execution is reflected in computing, data preprocessing, and distributed training. Unified deployment for all scenarios means that the framework supports cloud, edge, and device scenarios.

The following figure shows the overall MindSpore architecture:

![MindSpore-arch](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/beginner/images/introduction2.png)

- **Multi-domain Expansion**: Provide large model suite, domain suite, AI4S suite, provide users with usable-upon-unpacking models and functional interfaces, which are easy to use for R&D and reference realization based on the pre-built models of the suite.
- **Developer-Friendly**: MindExpression layer provides users with interfaces for AI model development, training, and inference, and supports users to develop and debug neural networks with native Python syntax, and its unique ability to unify dynamic and static graphs enables developers to take into account the development efficiency and execution performance, while the layer provides unified C++/Python interfaces for the whole scenario in the production and deployment phases.
- **Runtime-Efficient**：
    - Data processing (MindSpore Data): provides high-performance data loading, data preprocessing functions.
    - Computational graph construction (MindChute): provides a variety of composition mechanisms, supports the construction of computational graph translation based on Python AST, also supports the ability to build computational graphs based on Python bytecode.
    - Compiler Optimization (MindCompiler): the key module of the static graph model, mediated by the full-scenario unified intermediate expression (MindIR), compiles the front-end functions as a whole into the underlying language with higher execution efficiency, and at the same time performs global performance optimizations, including hardware-independent optimizations such as auto-differentiation and algebraic reduction, and hardware-relevant optimizations such as graph-operation fusion and operation generation.
    - Dynamic graph direct tuning: the key module of the dynamic graph model, based on the unified Python expression layer interface, matching Python interpreted execution mode, performing interface-wise interpreted execution. The reverse execution process reuses the unified automatic differentiation function.
- **Full-Scenario Deployment and Diversity Hardware**: The runtime (MindRT) connects and calls the underlying hardware operators according to the results of the upper-layer compilation and optimization, and supports "end-edge-cloud" AI collaboration including federated learning through the "end-edge-cloud" unified runtime architecture.
- **Others**: MindSpore Lite, an offline conversion tool and lightweight inference engine for lightweight inference, as well as debugging and tuning tools, MindSpore Armour, etc., are available for users to choose and use as needed.

### Execution Process

With an understanding of the overall architecture of MindSpore, we can look at the overall coordination relationship between the various modules, as shown in the figure:

![MindSpore](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/beginner/images/introduction4.png)

As an all-scenario AI framework, MindSpore supports different series of hardware in the device (mobile phone and IoT device), edge (base station and routing device), and cloud (server) scenarios, including Ascend series products and NVIDIA series products, Qualcomm Snapdragon in the ARM series, and Huawei Kirin chips.

The blue box on the left is the main MindSpore framework, which mainly provides the basic API functions related to the training and verification of neural networks, and also provides automatic differentiation and automatic parallelism by default.

Below the blue box is the MindSpore Data module, which can be used for data preprocessing, including data sampling, data iteration, data format conversion, and other data operations. Many debugging and tuning problems may occur during training. Therefore, the MindSpore Insight module visualizes debugging and tuning data such as the loss curve, operator execution status, and weight parameter variables, facilitating debugging and optimization during training.

The simplest scenario to ensure AI security is from the perspective of attack and defense. For example, attackers inject malicious data in the training phase to affect the inference capability of AI models. Therefore, MindSpore launches the MindSpore Armour module to provide an AI security mechanism for MindSpore.

The content above the blue box is closer to algorithm development users, including the AI algorithm model library ModelZoo, development toolkit MindSpore DevKit for different fields, and advanced extension library MindSpore Extend. MindSciences, a scientific computing kit in MindSpore Extend, is worth mentioning. MindSpore is the first to combine scientific computing with deep learning, combine numerical computing with deep learning, and support electromagnetic simulation and drug molecular simulation through deep learning.

After the neural network model is trained, you can export the model or load the model that has been trained in MindSpore Hub. Then MindIR provides a unified IR format for the device and cloud, which defines logical network structures and operator attributes through a unified IR, and decouples model files in MindIR format from hardware platforms to implement one-time training and multiple-time deployment. As shown in the figure, the model is exported to different modules through IR to perform inference.

### Design Philosophy

- Supporting unified deployment for all scenarios

    MindSpore is derived from industry-wide best practices. It provides unified model training, inference, and export APIs for data scientists and algorithm engineers. It supports flexible deployment in different scenarios such as the device, edge, and cloud, and promotes the prosperity of domains such as deep learning and scientific computing.

- Provideing the Python programming paradigm to simplify AI programming

    MindSpore provides a Python programming paradigm. Users can build complex neural network models using Python's native control logic, making AI programming easy.

- Providing a unified coding method for dynamic and static graphs

    Currently, there are two execution modes of a mainstream deep learning framework: a static graph mode (GRAPH_MODE) and a dynamic graph mode (PYNATIVE_MODE). The GRAPH mode has high training performance but is difficult to debug. On the contrary, the PYNATIVE mode is easy to debug, but is difficult to execute efficiently.
    MindSpore provides an encoding mode that unifies dynamic and static graphs, which greatly improves the compatibility between static and dynamic graphs. Instead of developing multiple sets of code, users can switch between the dynamic and static graph modes by changing only one line of code, which facilitates development and debugging, and improves performance experience.

    For example, set `set_context(mode=PYNATIVE_MODE)` to switch to the dynamic graph mode, or set `set_context(mode=GRAPH_MODE)` to switch to the static graph mode.

- Using AI and scientific computing fusion programming and allowing users to focus on the mathematical native expression of model algorithms

    On the basis of support for AI model training and inference programming, it extends the support for flexible automatic differential programming capability, supports differential derivation in the case of function and control flow expression, and supports various kinds of advanced differential capabilities, such as forward differentiation and higher-order differentiation, based on which users can realize the programming expression of differential functions commonly used in scientific computation, so as to support the fusion programming and development of AI and scientific computation.

- Distributed training native

    As a scale of neural network models and datasets continuously increases, parallel distributed training becomes a common practice of neural network training. However, the strategy selection and compilation of parallel distributed training are very complex, which severely restricts training efficiency of a deep learning model and hinders development of deep learning. MindSpore unifies the coding methods of single device and distributed training. Developers do not need to write complex distributed strategies. They can implement distributed training by adding a small amount of code to the single device code, which improves the efficiency of neural network training, greatly reduces the threshold of AI development, and enables users to quickly implement model ideas.

    For example, they can set `set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL)` to automatically establish a cost model, and select an optimal parallel mode for users.

### API Level Structure

MindSpore provides users with three different levels of APIs to support AI application (algorithm/model) development, from high to low: High-Level Python API, Medium-Level Python API and Low-Level Python API. The High-Level API provides better encapsulation, the Low-Level API provides better flexibility, and the Mid-Level API combines flexibility and encapsulation to meet the needs of developers in different fields and levels.

![MindSpore API](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/beginner/images/introduction3.png)

- High-Level Python API

    High-level APIs are at the first layer. Based on the medium-level API, these advanced APIs include training and inference management, mixed precision training, and debugging and optimization, enabling users to control the execution process of the entire network and implement training, inference, and optimization of the neural network. For example, by utilizing the Model API, users can specify the neural network model to be trained as well as related training settings, train the neural network model.

- Medium-Level Python API

    Medium-level APIs are at the second layer, which encapsulates low-cost APIs and provides such modules as the network layer, optimizer, and loss function. Users can flexibly build neural networks and control execution processes through the medium-level API to quickly implement model algorithm logic. For example, users can call the Cell API to build neural network models and computing logic, add the loss function and optimization methods to the neural network model by using the loss module and Optimizer API, and use the dataset module to process data for model training and derivation.

- Low-Level Python API

    Low-level APIs are at the third layer, including tensor definition, basic operators, and automatic differential modules, enabling users to easily define tensors and perform derivative computation. For example, users can customize tensors by using the Tensor API, and use the grad API to calculate the derivative of the function at a specified position.

## Introduction to Huawei Ascend AI Full-Stack Solution

Ascend computing is a full-stack AI computing infrastructure and application based on the Ascend series processors. It includes the Ascend series chips, Atlas series hardware, CANN chip enablement, MindSpore AI framework, ModelArts, and MindX application enablement.

Huawei Atlas AI computing solution is based on Ascend series AI processors and uses various product forms such as modules, cards, edge stations, servers, and clusters to build an all-scenario AI infrastructure solution oriented to device, edge, and cloud. It covers data center and intelligent edge solutions, as well as the entire inference and training processes in the deep learning field.

Th Ascend AI full stack is shown below:

![Ascend full stack](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/beginner/images/introduction1.png)

The functions of each module are described as follows:

- **Ascend Application Enablement**: AI platform or service capabilities provided by Huawei major product lines based on MindSpore.
- **MindSpore**: Support for device-edge-cloud-independent and collaborative unified training and inference frameworks.
- **CANN**: A driver layer that enables Ascend chips ([learn more](https://www.hiascend.com/en/software/cann)).
- **Compute Resources**: Ascend serialized IP, chips and servers.

For details, click [Huawei Ascend official website](https://e.huawei.com/en/products/servers/ascend).

## Joining the Community

Welcome every developer to the MindSpore community and contribute to this all-scenario AI framework.

- **MindSpore official website**: provides comprehensive MindSpore information, including installation, tutorials, documents, community, resources, and news ([learn more](https://www.mindspore.cn/en)).
- **MindSpore code**:

    - [MindSpore Gitee](https://gitee.com/mindspore/mindspore): Top 1 Gitee open-source project in 2020, where you can track the latest progress of MindSpore by clicking Watch, Star, and Fork, discuss issues, and commit code.

    - [MindSpore GitHub](https://github.com/mindspore-ai/mindspore): MindSpore code image of Gitee. Developers who are accustomed to using GitHub can learn MindSpore and view the latest code implementation here.

- **MindSpore forum**: We are dedicated to serving every developer. You can find your voice in MindSpore, regardless of whether you are an entry-level developer or a master. Let's learn and grow together. ([Learn more](https://www.hiascend.com/forum/forum-0106101385921175002-1.html))
