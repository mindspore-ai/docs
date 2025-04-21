# Third-Party Hardware Interconnection

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/design/pluggable_device.md)

MindSpore supports plug-in, standardized, low-cost and rapid interconnection of third-party chips through an open architecture:

- Decoupling of back-end architectures to quickly support plug-in interconnection of new chips.
- Modeling of abstract hardware types and standardization of interconnection processes.
- Abstract operator encapsulation, uniform selection of multi-chip heterogeneous operator.
- Support third-party graph IR access to give full play to the advantages of the chip architecture.

MindSpore overall architecture and components related to the backend are shown in the following figure:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindspore/source_en/design/images/pluggable_device_arch.png)

The overall MindSpore architecture consists of the following major components, which have interdependencies with each other:

- Python API: Provide a Python-based front-end expression and programming interface to support users in network construction, whole-graph execution, sub-graph execution and single-calculus execution, and call to C++ modules through the pybind11 interface, which are divided into front-end, back-end, MindData, and Core.
- MindExpression Front-End Expression: Responsible for compilation flow control and hardware-independent optimizations such as type derivation, automatic differentiation, expression simplification, etc.
- MindData Data Components: MindData provides efficient data processing, common dataset loading and other functions and programming interfaces, and supports users' flexible definition of processing registration and pipeline parallel optimization.
- MindIR: Contains ANF IR data structures, logs, exceptions, and other data structures and algorithms shared by end and cloud.

The process of third-party chip interconnection to MindSpore mainly involves the back-end of MindSpore, which is also divided into several components. The overall components are divided into two main categories:

- A category of hardware-independent components, commonly used data structures such as MemoryManager, MemoryPool, DeviceAddres and related algorithms as well as components including GraphCompiler, GraphSchdeduler that can schedule the entire process and have initial processing and scheduling capabilities for graphs or single operators.
- The other category is hardware-related components. This part provides several interfaces through the abstraction of hardware, and the third-party chips can choose interconnection according to the situation to realize the logic of operator, graph optimization, memory allocation, stream allocation, etc. unique to the hardware platform, and encapsulate them into dynamic libraries, which are loaded as plug-ins when the program runs. Third-party chips can refer to default built-in CPU/GPU/Ascend plug-ins of MindSpore when interconnection.

To facilitate third-party hardware interconnection, a hardware abstraction layer is provided in MindSpore, which defines a standardized hardware interconnection interface. The abstraction layer is called by two modules, GraphCompiler and GraphScheduler, in the upper unified runtime:

- GraphCompiler provides default control flow, heterogeneous graph splitting logic, graph optimization at different stages, calls operator selection/operator compilation, memory allocation and stream allocation provided by the abstraction layer.
- GraphScheduler is responsible for transforming the compiled graph into an Actor model and adding it to the thread pool, and executing the scheduling of these Actors.

Also public data structures and algorithms are provided in the framework, such as debug tools, default memory pool implementation, hundreds of common operations on Anf IR, and efficient memory reuse algorithm SOMAS developed by MindSpore.

The hardware abstraction layer provides Graph mode (GraphExecutor) and Kernel mode (KernelExecutor) for two interconnection methods, respectively, for DSA architecture (such as NPU, XPU) and general architecture chips (such as GPU, CPU) to provide a classified interconnection interface. Chip vendors can inherit one or two abstract classes and implement them. Depending on the interconnection method, if you interconnect to Kernel mode, you also need to implement DeviceResMananger, KernelMod, DeviceAddress and other interfaces.

## Kernel Mode Interconnection

The generic architecture Kernel mode requires the following aspects to be implemented in the plug-in:

- Custom graph splitting logic, which allows for low-cost implementation of the control flow, heterogeneity and other advanced features provided by the framework. It can be null implementation if the features are not used.

- Custom graph optimization, which allows splitting and fusion of certain operators according to the features of the hardware, and other custom modifications to the graph.

- Operator selection and operator compilation.
- Memory management. DeviceAddres is the abstraction of memory, and third-party chip vendors need to implement the function of copying between Host and Device. It also needs to provide memory request and destruction functions. To facilitate third-party chip vendors, MindSpore provides a set of memory pool implementations and an efficient memory reuse algorithm, SOMAS, in the Common component.
- Stream management. If the chip to be docked has the concept of stream, it needs to provide the function of creation and destruction. and If not, it will run in single stream mode.

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindspore/source_zh_cn/design/images/pluggable_device_kernel.png)

## Graph Mode Interconnection

If the chip vendor's software stack can provide completely high level APIs, or if there are differences between the software stack of DSA architecture chips and Kernel mode, it can interconnect to Graph mode. The Graph model treats the whole graph as a big operator (SuperKernel) implemented by a third-party software stack, which needs to implement the following two functions:

- Graph compilation. The third-party chip vendor needs to transform MindSpore Anf IR into a third-party IR graph representation and perform a third-party graph compilation process to compile the graph to an executable ready state.

- Graph execution. The third-party chip vendor needs to understand MindSpore Tensor format or transform it into a format that can be understood, and call the execution of the ready graph and transform the result of the execution into MindSpore Tensor format.

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindspore/source_zh_cn/design/images/pluggable_device_graph.png)
