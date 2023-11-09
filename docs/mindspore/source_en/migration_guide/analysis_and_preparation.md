# Model Analysis and Preparation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/migration_guide/analysis_and_preparation.md)

## Reproducing Algorithm Implementation

1. Obtain the PyTorch reference code.
2. Analyzing the algorithm, network structure, and tricks in the original code, including the method of data augmentation, learning rate attenuation policy, optimizer parameters, and the initialization method of training parameters, etc.
3. Reproduce the accuracy of the reference implementation, obtain the performance data of the reference implementation, and identify some issues in advance.

Please refer to [Details of Reproducing Algorithm Implementation](https://www.mindspore.cn/docs/en/r2.3/migration_guide/reproducing_algorithm.html).

## Analyzing API Compliance

Before practicing migration, it is recommended to analyze the API compliance in MindSpore's migration code to avoid affecting code implementation due to the lack of API support.

The API missing analysis here refers to APIs in the network execution diagram, including MindSpore [operators](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore.ops.primitive.html) and advanced encapsulated APIs, and excluding the APIs used in data processing. You are advised to use third-party APIs, such as NumPy, OpenCV, Pandas, and PIL, to replace APIs used in data processing.

There are two methods to analyze API compliance:

1. Scanning API by MindSpore Dev Toolkit.
2. Querying the API Mapping Table.

### Scanning API by Toolkit

[MindSpore Dev Toolkit](https://www.mindspore.cn/devtoolkit/docs/en/master/index.html) is a development kit supporting PyCharm and Visual Studio Code plug-in developed by MindSpore, which can scan API based on file-level or project-level.

Refer to [PyCharm API Scanning](https://www.mindspore.cn/devtoolkit/docs/en/master/api_scanning.html) for the tutorials of Dev Toolkit in PyCharm.

![api_scan_pycharm](./images/api_scan_pycharm.jpg)

Refer to [Visual Studio Code API Scanning](https://www.mindspore.cn/devtoolkit/docs/en/master/VSCode_api_scan.html) for the tutorials of Dev Toolkit in PyCharm.

![api_scan_pycharm](./images/api_scan_vscode.jpg)

### Querying the API Mapping Table

Take the PyTorch code migration as an example. After obtaining the reference code implementation, you can filter keywords such as `torch`, `nn`, and `ops` to obtain the used APIs. If the method of another repository is invoked, you need to manually analyze the API. Then, check the [PyTorch and MindSpore API Mapping Table](https://www.mindspore.cn/docs/en/r2.3/note/api_mapping/pytorch_api_mapping.html).
Alternatively, the [API](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore.ops.primitive.html) searches for the corresponding API implementation.

For details about the mapping of other framework APIs, see the [API naming and function description](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore.html). For APIs with the same function, the names of MindSpore may be different from those of other frameworks. The parameters and functions of APIs with the same name may also be different from those of other frameworks. For details, see the official description.

### Process Missing API

You can use the following methods to process the missing API:

1. Use equivalent replacement
2. Use existing APIs to package equivalent function logic
3. Customize operators
4. Seek help from the community

Refer to [Missing API Processing Policy](https://www.mindspore.cn/docs/en/r2.3/migration_guide/missing_api_processing_policy.html) for details.

## Analyzing Function Compliance

During continuous delivery of MindSpore, some functions are restricted. If restricted functions are involved during network migration, some measures can be taken to avoid the impact of function restrictions, such as [dynamic shape](https://www.mindspore.cn/docs/en/r2.3/migration_guide/dynamic_shape.html) and [sparsity](https://www.mindspore.cn/docs/en/r2.3/migration_guide/sparsity.html), etc.

## MindSpore Function and Feature Recommendation

### [Dynamic and Static Graphs](https://www.mindspore.cn/tutorials/en/r2.3/beginner/accelerate_with_static_graph.html)

Currently, there are two execution modes of a mainstream deep learning framework: a static graph mode (Graph) and a dynamic graph mode (PyNative).

- In static graph mode, when the program is built and executed, the graph structure of the neural network is generated first, and then the computation operations involved in the graph are performed. Therefore, in static graph mode, the compiler can achieve better execution performance by using technologies such as graph optimization, which facilitates large-scale deployment and cross-platform running.

- In dynamic graph mode, the program is executed line by line according to the code writing sequence. In the forward execution process, the backward execution graph is dynamically generated according to the backward propagation principle. In this mode, the compiler delivers the operators in the neural network to the device one by one for computing, facilitating users to build and debug the neural network model.

### [Calling the Custom Class](https://www.mindspore.cn/tutorials/en/r2.3/advanced/static_graph_expert_programming.html#using-jit-class)

In static graph mode, you can use `jit_class` to modify a custom class. You can create and call an instance of the custom class, and obtain its attributes and methods.

`jit_class` is applied to the static graph mode to expand the support scope of static graph compilation syntax. In dynamic graph mode, that is, PyNative mode, the use of `jit_class` does not affect the execution logic of PyNative mode.

### [Automatic Differential](https://www.mindspore.cn/tutorials/en/r2.3/beginner/autograd.html)

Automatic differentiation can calculate a derivative value of a derivative function at a certain point, which is a generalization of backward propagation algorithms. The main problem solved by automatic differential is to decompose a complex mathematical operation into a series of simple basic operations. This function shields a large number of derivative details and processes from users, greatly reducing the threshold for using the framework.

### [Mixed Precision](https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/mixed_precision.html)

Generally, when a neural network model is trained, the default data type is FP32. In recent years, to accelerate training time, reduce memory occupied during network training, and store a trained model with same precision, more and more mixed-precision training methods are proposed in the industry. The mixed-precision training herein means that both single precision (FP32) and half precision (FP16) are used in a training process.

### [Auto Augmentation](https://www.mindspore.cn/tutorials/experts/en/r2.3/dataset/augment.html)

MindSpore not only allows you to customize data augmentation, but also provides an automatic data augmentation mode to automatically perform data augmentation on images based on specific policies.

### [Gradient Accumulation Algorithm](https://www.mindspore.cn/tutorials/experts/en/r2.3/optimize/gradient_accumulation.html)

Gradient accumulation is a method of splitting data samples for training neural networks into several small batches by batch and then calculating the batches in sequence. The purpose is to solve the out of memory (OOM) problem that the neural network cannot be trained or the network model cannot be loaded due to insufficient memory.

### [Summary](https://www.mindspore.cn/mindinsight/docs/en/master/summary_record.html)

Scalars, images, computational graphs, training optimization processes, and model hyperparameters during training are recorded in files and can be viewed on the web page.

### [Debugger](https://www.mindspore.cn/mindinsight/docs/en/master/debugger.html)

The MindSpore debugger is a debugging tool provided for graph mode training. It can be used to view and analyze the intermediate results of graph nodes.

### [Golden Stick](https://www.mindspore.cn/golden_stick/docs/en/master/index.html)

MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei Noah's team and Huawei MindSpore team. It contains basic quantization and pruning methods.

## Differences Between MindSpore and PyTorch APIs

When migrating the network from PyTorch to MindSpore, pay attention to the differences between MindSpore.
