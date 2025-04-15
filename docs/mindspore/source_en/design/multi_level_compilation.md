# Multi-Level Compilation Architecture

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/multi_level_compilation.md)

## Background

With the arrival of the era of deep learning large models, the bigger the network size is,  the bigger the challenge of graph compilation performance, execution performance and debugging and tuning efficiency is. For this reason, MindSpore proposes a multilevel compilation architecture that provides an O(n) multilevel compilation execution model, which are different from each other in terms of graph optimization, operator fusion, memory management, and execution modes, and is designed to provide a diversity of graph mode. Users can choose the most suitable compilation and execution mode according to their own network characteristics and needs:

1. O0 mode: this is a basic compilation and execution mode, where all optimizations are turned off except those necessary to affect the functionality, and a single-calculus execution is used for execution. Therefore, the execution performance may not be optimal, but it can guarantee the original structure of the graph, which is convenient for users to debug and understand, and the compilation performance is also better. Add and Mul single operator execution is shown in the following figure.
2. O1 mode: this mode performs some basic optimizations, such as common graph optimization and automatic operator fusion optimization, and uses single operator execution for execution. Compared with O0, because of enabling the fusion optimization, the execution performance of O1 can be improved, but it may affect the original structure of the graph, so the compilation performance and debugging and tuning efficiency is lost. In the following figure, Add and Mul are fused into a single fused_op execution.
3. O2 mode: this is a more advanced optimization mode, currently not implemented, the subsequent deeper optimization can use this mode.

![jit_level_example](./images/multi_level_compilation/jit_level_example.png)

## Overview of Multi-Level Compilation Architecture

![jit_level_framework](./images/multi_level_compilation/jit_level_framework.png)

1. Multi-level compilation external interface: configure multi-level compilation level through [mindspore.jit(jit_level=“O0/O1”)](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html#mindspore.jit), jit_level defaults to O0. We usually recommend that users use O0 mode for network debugging tuning. After debugging is ready, for better performance you can turn on O1 to run the network.
2. Backend graph compilation: According to the configured multi-level compilation level, different compilation modes are selected. O0 is the most basic native composition and compilation, and O1 adds automatic operator fusion function on the basis of O0, with the main functions of graph optimization, graph-operator fusion, operator selection, and execution sequence scheduling, of which graph-operator fusion is a unique function in O1 mode.
3. Backend graph execution: The O0 and O1 modes are the same at the execution level, and both use a single operator way of scheduling execution, with the main functions of multi-stream concurrency, multi-level streaming, HAL management, and memory management.

## Introduction to the O0 Model

O0 is the basic graph compilation and execution mode, except for the necessary impact on the functionality of the optimization, other optimizations are turned off, the use of native graph structure for compilation and execution, easy to debug and tuning, with better compilation performance. The following mainly introduces the functions related to backend graph compilation, and the functions related to backend graph execution are detailed in [runtime](https://www.mindspore.cn/docs/en/master/features/runtime/memory_manager.html).

### Graph Optimization

There are fewer graph optimizations for the O0 mode, and the basic optimizations are mainly back-end LazyInline and No-task node execution optimizations.

- **Back-end LazyInline**

  **LazyInline**: The main idea is to postpone the overhead of the function call to the actual need to call , so that you can reduce the compilation overhead, improve compilation efficiency.LazyInline is the same sub-graph structure reuse in the graph compilation phase, do not unfolding placed in the graph , to avoid the graph size is large resulting in the impact of the compilation performance.

  ![jit_level_lazyinline](./images/multi_level_compilation/jit_level_lazyinline.png)

  **Pipeline Parallelism**: Slicing the operator in the neural network into multiple Stages, and then mapping the Stages to different devices, so that different devices to compute different parts of the neural network. In order to improve efficiency, pipeline parallelism further slices the MiniBatch into finer-grained MicroBatches, in which pipelined scheduling is used, thus achieving the goal of improving efficiency.

  **Back-end LazyInline**: Since MicroBatch slicing of Pipeline parallel leads to the expansion of the entire computational graph to a number of times of the MicroBatch, which results in a huge model size and long compilation performance time (possibly hour-level), and these Micro subgraphs are all structured the same way. In order to solve the compilation performance problem, the LazyInline technique is a great fit, however LazyInline brings problems such as inability to use the optimal way for memory reuse and stream allocation at runtime, inability to perform cross-graph optimization (memory optimization, communication fusion, operator fusion, etc.). For this reason, at the end of the compilation of the graph, before the execution of the graph, these Micro subgraphs are as the actual nodes of Inline in order to form a complete global whole graph, and then through memory optimization, communication optimization, redundant computation elimination after the graph Inline, so as to achieve the goal of compilation performance, execution performance, and execution memory are taken into account.

- **No-task node Execution Optimization**

  ![jit_level_no_task](./images/multi_level_compilation/jit_level_no_task.png)

  No-task node refers to Reshape, ExpandDims, Squeeze, Flatten, FlattenGrad, Reformat, etc. There is no computational logic in these algorithms, and they do not modify the memory layout, but only modify the information of the shape, format. At the end of the compilation of the graph, the No-task node is converted to ref node, the output has the same address as the input, and the kernel launch is skipped in the execution process, so as to achieve the purpose of execution performance optimization.

### Operator Selection

Operators are the basic execution units in deep learning frameworks, and they are responsible for performing specific computational tasks, such as matrix multiplication, convolution, pooling. Operator selection requires comprehensive consideration of factors such as operator type, data type, hardware platform, and operator optimization in order to select the optimal operator for deep learning tasks.

The operator types in the backend of MindSpore Ascend are Aclnn kernel/Aclop kernel/Hccl kernel /Cpu kernel, and the process of operator selection is shown as follows:

![jit_level_kernelselect](./images/multi_level_compilation/jit_level_kernelselect.png)

1. operator type: firstly, according to the type of operator, choose whether it is computational operator or communication operator.
2. hardware platform: If there is a corresponding operator on hardware, then the operator on hardware is preferred, otherwise the heterogeneous operator on CPU is chosen, e.g., shape-related computational operators may only be suitable to be supported on CPU, and there is no corresponding operator on hardware.
3. operator efficiency: due to the better performance of Aclnn operator on Ascend, the computational operator will prefer Aclnn kernel if there is a corresponding Aclnn kernel, otherwise Aclop kernel will be chosen.
4. If no operator is selected in any of the above 3 steps, it is an unsupported operator and the operator selection fails to exit.

### Executing Order Organization

![jit_level_exec_order](./images/multi_level_compilation/jit_level_exec_order.png)

Different graph traversal algorithms produce execution orders with large differences in execution performance and memory, as shown in the figure above:

- **Execution order obtained by BFS**: kernel1-> kernel2-> kernel4-> kernel5-> kernel3-> kernel6. Memory peaks at 5G (kernel3 can release kernel1 and kernel2 after execution, and then reuse them when it's kernel6's turn to execute, so kernel6 doesn't need to request extra memory).
- **Execution order obtained by DFS**: kernel1-> kernel2-> kernel3-> kernel4-> kernel5-> kernel6. Memory peaks at 4G (kernel3 can release kernel1 and kernel2 after execution, and then reuse them when it's kernel4 and kernel5's turn to execute, so kernel4 and kernel5 don't need to request extra memory).

Execution order scheduling is a complex problem of solving optimal operator concurrency under certain memory constraints, which not only requires identifying and exploiting concurrency opportunities in the computational graph to improve computational efficiency, but also must consider multiple constraints at the same time to ensure the stability and efficiency of the system.

- First, the optimization module needs to address the complexity of solving for optimal operator concurrency. Due to the large number of operators in the computational graph and their interdependencies, finding an execution order that maximizes concurrency while maintaining the logical correctness of the computational graph is a challenging task.
- Second, memory constraints are a critical factor that cannot be ignored in execution order optimization. Increasing concurrency, while improving computational efficiency, tends to significantly increase peak memory requirements, which may lead to Overflow of Memory (OOM) errors, especially in resource-constrained environments. Therefore, the optimization module must weigh the relationship between concurrency and memory usage to ensure that concurrency is increased without exceeding the memory capacity of the system.
- MindSpore's execution order adjustment module combines rule-based and heuristic-based strategies to provide both bfs/dfs execution order orchestration algorithms [mindspore.jit(option={“exec_order”: “bfs/dfs”})](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html#mindspore.jit) to achieve fine-grained adjustment of the execution order of the computation graph, so as to effectively deal with multiple challenges such as memory constraints and system stability while ensuring computational efficiency.

## Introduction to the O1 Model

O1 is mainly targeted at implementing general-purpose, generalizable AI compilation optimizations on top of O0 to support better execution performance requirements for most general-purpose training and inference scenarios.

In the current phase, O1 mainly supports graph-kernel fusion optimization. The main idea is to automatically identify neighboring fusable nodes in the computational graph during the static graph compilation phase, and then fuse them into executable operators with larger granularity. Through graph-kernel fusion, optimization effects such as increasing the computational locality of operators and reducing the overall global memory access bandwidth overhead are achieved. As verified by real-world tests on 15+ networks, O1 is able to achieve an average of 15% performance acceleration compared to O0. Especially for access-intensive networks, the optimization effect of O1 is more significant.

### Graph-Kernel Fusion

Mainstream AI computing frameworks such as MindSpore provide operators to users that is usually defined in terms of understandable and easy use for user. Each operator carries a different amount of computation and varies in computational complexity. However, from the hardware execution point of view, this natural, user perspective-based division of operator computation volume is not efficient and does not fully utilize the computational power of hardware resources, which is mainly reflected in the following aspects:

1. Computationally overloaded and overly complex operators, which usually makes it difficult to generate well-cut high-performance operator, thereby reducing equipment utilization.
2. Operators that are too small in computation may also cause latency in computation and thus reduce equipment utilization, as the computation cannot effectively hide the data moving overhead.
3. Hardware Devices are usually multi-core, many-core architectures. When the operator shape is small or other reasons cause insufficient computational parallelism, it may cause some cores to be idle, thus reducing the device utilization. In particular, chips based on Domain Specific Architecture (DSA for short) are more sensitive to these factors. It has been a big challenge to maximize the performance of hardware operator while making the operator easy to use.

In terms of AI framework design, the current industry mainstream adopts a separate layer implementation approach of graph and operator layers. The graph layer is responsible for fusing or regrouping the computational graph, and the operator layer is responsible for compiling the fused or regrouped operators into high-performance executable operators. The graph layer is usually processed and optimized by using Tensor-based High-Level IR, while the operator layer is analyzed and optimized by using computational instruction-based Low-Level IR. This artificial separate-layer process significantly increases the difficulty of performing collaborative optimization in both graph and computational layers.

MindSpore has adopted the technique of graph-kernel fusion to better solve this problem in the past few years. Typical networks in different categories such as NLP and recommendation show significant gains in training speed after enabling graph-kernel fusion. One of the main reasons is the presence of a large number of small operator combinations in these networks, which have more opportunities for fusion optimization.

#### Graph-Kernel Fusion Architecture and Overall Process

The overall architecture of graph-kernel fusion is shown in the figure below. The main idea in the graph layer is to turn on the composite operator, then perform cross-boundary aggregation and optimization, and finally perform Kernel operator splitting. The main steps include:

1. Composite Expansion: Expand the composite operator into the basic operator and form the Composite subgraph to facilitate subsequent cross-boundary optimization and operator splitting.
2. Cross-OP Aggregation: Aggregate adjacent elementary operators or Composite subgraphs to form larger aggregated subgraphs for subsequent cross-boundary optimization and operator splitting.
3. High-Level Optimization: Based on the aggregated subgraphs obtained in the above two steps, we can perform a large number of cross-boundary optimizations, such as algebraic simplification, common subexpression extraction (CSE).
4. Kernel Partition: Based on the computational features and the performance of the fusion operator, the operator splitting is performed on the aggregated computational subgraph.

The optimized computational graph is passed to MindSpore AKG as a subgraph for further back-end optimization and target code generation.

![graphkernel](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/graphkernel.png)

By following these steps, we can obtain two aspects of performance gains:

1. Cross-boundary performance optimization gains between different operators.
2. The optimal granularity of the fusion operator is obtained by reorganizing and splitting the entire computational graph.

#### Fusion Operator Acceleration Optimization (MindSpore AKG)

As mentioned earlier, in scenarios such as HPC and deep neural network training, graph-kernel fusion optimization can bring exponential performance improvements. However, with the increasing capability of graph-kernel fusion, the development of fusion operator becomes a bottleneck point to continue to improve the graph-kernel fusion capability. The automatic generation technology of fusion operators can solve the problem of high programming threshold for developing fusion operators based on DSA, allowing programmers to focus on the implementation logic of operators during operator development without focusing on back-end optimization, which greatly improves their development efficiency. Especially for scenarios with complex back-end hardware architectures and the presence of complex operators and fusion operators, automatic operator generation techniques are more critical.

Therefore, **MindSpore AKG accelerates optimization and automatic generation of fusion operator based on Polyhedral Compilation Technology (Polyhedral Model)**, can help fused operators optimized by MindSpore graph-kernel fusion module to automatically generate high-performance kernel on **heterogeneous hardware platforms** (GPU/Ascend) and improve MindSpore training performance.

Architecture and Overall Process:

The overall framework of MindSpore AKG is shown in the figure above:

- IR Normalization
    - The input of MindSpore AKG is the fused subgraph optimized by MindSpore graph-kernel fusion module, and the operator in the subgraph is expressed by various descriptions such as TVM's Compute/IR Builder/Hybrid. The DSL is then converted to Halide IR ([Halide](https://halide-lang.org/), a common language used to develop high-performance image processing and Array computation, which can be used as an intermediate expression for decoupling algorithms and optimization) and IR normalization.
    - After the initial simplification and optimization is completed, the Halide IR is transformed into the scheduling tree required by the Poly module.
- Poly module scheduling optimization
    - Using the Pluto scheduling algorithm in Polyhedral technology to achieve automatic fusion of loops, automatic rearrangement and other transformations to automatically generate an initial schedule that satisfies parallelism and data locality for the fusion operator.
    - To quickly adapt to different hardware backends, the optimization pass in the Poly module is divided into hardware-independent generic optimizations and hardware-related specific optimizations, which are stitched and combined according to hardware features at compilation time, to achieve fast adaptation of heterogeneous hardware backends. The pass such as Auto-slicing, auto-mapping and auto-memory boosting will give different optimizations depending on the nature of the hardware architecture.
- Backends optimization
    - In order to further improve the performance of the operator, we developed corresponding optimization passes for different hardware backends, such as data alignment and instruction mapping in Ascend backend, vectorized access and insertion of synchronization instructions in GPU backend, and finally generated the corresponding platform code.

### other graph optimization techniques

In addition to graph-kernel fusion, O1 may be gradually extended to add some other graph optimization techniques in subsequent releases. For example:

1. KernelPacket: automatic fusion and optimization of shape computations in dynamic shape scenarios;
2. Communicative-kernel fusion: fusion of communication operators with computational operators.
