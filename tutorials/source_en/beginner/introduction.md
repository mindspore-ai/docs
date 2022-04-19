# Overview

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/introduction.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following describes the Huawei AI full-stack solution and the position of MindSpore in the solution. Developers who are interested in MindSpore can visit the [MindSpore community](https://gitee.com/mindspore/mindspore) and click [Watch, Star, and Fork](https://gitee.com/mindspore/mindspore).

## Introduction to MindSpore

MindSpore is a deep learning framework in all scenarios, aiming to achieve easy development, efficient execution, and all-scenario coverage.

Easy development features user-friendly APIs and low debugging difficulty. Efficient execution is reflected in computing, data preprocessing, and distributed training. All-scenario coverage means that the framework supports cloud, edge, and device scenarios.

The following figure shows the overall MindSpore architecture:

![MindSpore-arch](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/beginner/images/introduction2.png)

- **ModelZoo**: provides available deep learning algorithm networks, and more developers are welcome to contribute new networks. ([ModelZoo](https://gitee.com/mindspore/models))
- **Extend**: The MindSpore expansion package supports new domain scenarios, such as GNN, deep probabilistic programming, and reinforcement learning. More developers are expected to contribute and build the library.
- **Science**: MindScience is a scientific computing kit for various industries based on the converged MindSpore framework. It contains the industry-leading datasets, basic network structures, high-precision pre-trained models, and pre-and post-processing tools that accelerate application development of the scientific computing ([More Information](https://mindspore.cn/mindscience/docs/en/master/index.html)).
- **Expression**: Python-based frontend expression and programming interfaces. In the future, Huawei plans to continue to provide interconnection with third-party front-end systems, such as C/C++ and Huawei-developed programming language front-end (currently in the pre-research phase), to introduce more third-party ecosystems.
- **Data**: provides functions such as efficient data processing, common dataset loading and programming interfaces, and allows users to flexibly define processing registration and pipeline parallel optimization.
- **Compiler**: The core compiler of the layer, which implements three major functions based on the unified device-cloud MindIR, including hardware-independent optimization (type derivation, automatic differentiation, and expression simplification), hardware-related optimization (automatic parallelism, memory optimization, graph kernel fusion, and pipeline execution), and optimization related to deployment and inference (quantification and pruning).
- **Runtime**: MindSpore runtime system, including the runtime system on the cloud host, runtime system on the device, and lightweight runtime system of the IoT platform.
- **Insight**: MindSpore visualized debugging and tuning tool, allowing users to debug and tune the training network ([More Information](https://mindspore.cn/mindinsight/docs/en/master/index.html)).
- **Armour**: For enterprise-level applications, provides enhanced functions related to security and privacy protection, such as anti-robustness, model security testing, differential privacy training, privacy leakage risk assessment, and data drift detection. ([Learn more] (<https://mindspore.cn/mindarmour/docs/en/master/index.html>))

### Execution Process

With an understanding of the overall architecture of MindSpore, we can look at the overall coordination relationship between the various modules, as shown in the figure:

![MindSpore](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/beginner/images/introduction4.png)

As an all-scenario AI framework, MindSpore supports different series of hardware in the device (mobile phone and IoT device), edge (base station and routing device), and cloud (server) scenarios, including Ascend series products and NVIDIA series products, Qualcomm Snapdragon in the ARM series, and Huawei Kirin chips.

The blue box on the left is the main MindSpore framework, which mainly provides the basic API functions related to the training and verification of neural networks, and also provides automatic differentiation and automatic parallelism by default.

Below the blue box is the MindSpore Data module, which can be used for data preprocessing, including data sampling, data iteration, data format conversion, and other data operations. Many debugging and tuning problems may occur during training. Therefore, the MindSpore Insight module visualizes debugging and tuning data such as the loss curve, operator execution status, and weight parameter variables, facilitating debugging and optimization during training.

The simplest way to ensure AI security is from the perspective of attack and defense. For example, attackers inject malicious data in the training phase to affect the inference capability of AI models. Therefore, MindSpore launches the MindSpore Armour module to provide an AI security mechanism for MindSpore.

The content above the blue box is closer to algorithm development users, including the AI algorithm model library ModelZoo, development toolkit MindSpore DevKit for different fields, and advanced extension library MindSpore Extend. MindSciences, a scientific computing kit in MindSpore Extend, is worth mentioning. MindSpore is the first to combine scientific computing with deep learning, combine numerical computing with deep learning, and support electromagnetic simulation and drug molecular simulation through deep learning.

After the neural network model is trained, you can export the model or load the model that has been trained in MindSpore Hub. Then MindIR provides a unified IR format for the device and cloud, which defines logical network structures and operator attributes through a unified IR, and decouples model files in MindIR format from hardware platforms to implement one-time training and multiple-time deployment. As shown in the figure, the model is exported to different modules through IR to perform inference.

### Design Philosophy

- Supporting full-scenario collaboration

    MindSpore is an industry-wide best practice. It provides unified model training, inference, and export APIs for data scientists and algorithm engineers. It supports flexible deployment in different scenarios such as the device, edge, and cloud, and promotes the prosperity of domains such as deep learning and scientific computing.

- Provideing the Python programming paradigm to simplify AI programming

    MindSpore provides a Python programming paradigm. Users can build complex neural network models using Python's native control logic, making AI programming easy.

- Providing a unified coding method for dynamic and static graphs

    Currently, there are two execution modes of a mainstream deep learning framework: a static graph mode (GRAPH_MODE) and a dynamic graph mode (PYNATIVE_MODE). The GRAPH mode has high training performance but is difficult to debug. On the contrary, the PYNATIVE mode is easy to debug, but is difficult to execute efficiently.
    MindSpore provides an encoding mode that unifies dynamic and static graphs, which greatly improves the compatibility between static and dynamic graphs. Instead of developing multiple sets of code, users can switch between the dynamic and static graph modes by changing only one line of code. For example, set `context.set_context(mode=context.PYNATIVE_MODE)` to switch to the dynamic graph mode, or set `context.set_context(mode=context.GRAPH_MODE)` to switch to the static graph mode, which facilitates development and debugging, and improves performance experience.

- Using functional differentiable programming architecture and allowing users to focus on the mathematical native expression of model algorithms

    Neural network models are usually trained based on gradient descent algorithms, but the manual derivation process is complicated and the results are prone to errors. The Automatic Differentiation mechanism based on Source Code Transformation (SCT) of MindSpore adopts a functional differentiable programming architecture, and provides a Python programming interface at the interface layer, including the expression of control flow. Users can focus on the mathematically native expression of the model algorithm without manual derivation.

- Unifying the coding method of single device and distributed training

    As a scale of neural network models and datasets continuously increases, parallel distributed training becomes a common practice of neural network training. However, the strategy selection and compilation of parallel distributed training are very complex, which severely restricts training efficiency of a deep learning model and hinders development of deep learning. MindSpore unifies the coding methods of single device and distributed training. Developers do not need to write complex distributed strategies. They can implement distributed training by adding a small amount of code to the single device code. For example, they can set `context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL)` to automatically establish a cost model, select an optimal parallel mode for users, improve the efficiency of neural network training, greatly reduce the threshold of AI development, and enable users to quickly implement model ideas.

### API Level Structure

To support network building, entire graph execution, subgraph execution, and single-operator execution, MindSpore provides users with three levels of APIs. In ascending order, these are Low-Level Python API, Medium-Level Python API, and High-Level Python API.

![MindSpore API](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/introduction3.png)

- High-Level Python API

    High-level APIs are at the first layer. Based on the medium-level API, these advanced APIs include training and inference management, mixed precision training, and debugging and optimization, enabling users to control the execution process of the entire network and implement training, inference, and optimization of the neural network. For example, by utilizing the Model API, users can specify the neural network model to be trained as well as related training settings, train the neural network model, and debug the neural network performance through the Profiler API.

- Medium-Level Python API

    Medium-level APIs are at the second layer, which encapsulates low-cost APIs and provides such modules as the network layer, optimizer, and loss function. Users can flexibly build neural networks and control execution processes through the medium-level API to quickly implement model algorithm logic. For example, users can call the Cell API to build neural network models and computing logic, add the loss function and optimization methods to the neural network model by using the loss module and Optimizer API, and use the dataset module to process data for model training and derivation.

- Low-Level Python API

    Low-level APIs are at the third layer, including tensor definition, basic operators, and automatic differential modules, enabling users to easily define tensors and perform derivative computation. For example, users can customize tensors by using the Tensor API, and use the GradOperation operator in the ops.composite module to calculate the derivative of the function at a specified position.

## Introduction to Huawei Ascend AI Full-Stack Solution

Ascend computing is a full-stack AI computing infrastructure and application based on the Ascend series processors. It includes the Ascend series chips, Atlas series hardware, CANN chip enablement, MindSpore AI framework, ModelArts, and MindX application enablement.

Huawei Atlas AI computing solution is based on Ascend series AI processors and uses various product forms such as modules, cards, edge stations, servers, and clusters to build an all-scenario AI infrastructure solution oriented to device, edge, and cloud. It covers data center and intelligent edge solutions, as well as the entire inference and training processes in the deep learning field.

The following figure shows the Ascend AI full-stack architecture:

![ Ascend Full Stack](https://gitee.com/mindspore/docs/raw/tutorials-develop/tutorials/source_en/beginner/images/introduction1.png)

The functions of each module are described as follows:

- **Atlas series**: provides AI training, inference cards, and training servers ([learn more](https://e.huawei.com/en/products/cloud-computing-dc/atlas/)).
- **CANN at heterogeneous computing architecture**: a driver layer that enables chips ([learn more](https://www.hiascend.com/en/software/cann)).
- **MindSpore**: all-scenario AI framework ([learn more](https://www.mindspore.cn/en)).
- **MindX SDK (Ascend SDK)**: provides the application solution ([learn more](https://www.hiascend.com/en/software/mindx-sdk)).
- **ModelArts**: HUAWEI CLOUD AI development platform ([learn more](https://www.huaweicloud.com/product/modelarts.html)).
- **MindStudio**: E2E development toolchain that provides one-stop IDE for AI development ([learn more](https://www.hiascend.com/en/software/mindstudio)).

For details, click [Huawei Ascend official website](https://e.huawei.com/en/products/servers/ascend).

## Joining the Community

Welcome every developer to the MindSpore community and contribute to this all-scenario AI framework.

- **MindSpore official website**: provides comprehensive MindSpore information, including installation, tutorials, documents, community, resources, and news ([learn more](https://www.mindspore.cn/en)).
- **MindSpore code**:

    - [MindSpore Gitee](https://gitee.com/mindspore/mindspore): Top 1 Gitee open-source project in 2020, where you can track the latest progress of MindSpore by clicking Watch, Star, and Fork, discuss issues, and commit code.

    - [MindSpore Github](https://github.com/mindspore-ai/mindspore): MindSpore code image of Gitee. Developers who are accustomed to using GitHub can learn MindSpore and view the latest code implementation here.

- **MindSpore forum**: We are dedicated to serving every developer. You can find your voice in MindSpore, regardless of whether you are an entry-level developer or a master. Let's learn and grow together. ([Learn more](https://bbs.huaweicloud.com/forum/forum-1076-1.html))
