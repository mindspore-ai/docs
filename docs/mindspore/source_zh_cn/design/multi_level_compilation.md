# 多级编译架构

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/mindspore/source_zh_cn/design/multi_level_compilation.md)

## 背景

随着深度学习大模型时代的到来，网络规模越来越大，对图编译性能、执行性能和调试调优效率的挑战也越来越大。为此，MindSpore提出多级编译架构，提供O(n)多级编译执行模式，它们在图优化、算子融合、内存管理以及执行模式等方面有所不同，旨在提供图模式的多样性选择，用户可以根据自己的网络特点和需求，选择最适合的编译执行模式：

1. O0模式：这是一个基础的编译执行模式，除必要影响功能的优化外，其他优化均关闭，使用单算子执行的执行方式。因此执行性能可能不是最优，但它的优点是可以保证图的原始结构，方便用户进行调试和理解，编译性能也较好。如下图中的Add和Mul单算子执行。
2. O1模式：这个模式会进行一些基础的优化，比如常用图优化和自动算子融合优化，使用单算子执行的执行方式。相比O0，由于使能了融合优化，可以提高执行性能，但可能会影响到图的原始结构，因此编译性能和调试调优效率有所损失。如下图中的Add跟Mul融合成一个fused_op执行。
3. O2模式：这是一个更高级的优化模式，目前没有实现，后续较为深层次的优化可使用该模式。

![jit_level_example](./images/multi_level_compilation/jit_level_example.png)

## 多级编译架构概述

![jit_level_framework](./images/multi_level_compilation/jit_level_framework.png)

1. 多级编译对外接口：通过[mindspore.jit(jit_level="O0/O1")](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.jit.html#mindspore.jit)来配置多级编译级别，jit_level默认为O0，通常我们建议用户使用O0模式进行网络调试调优，调试就绪后，为了更好的性能可以一键开启O1运行网络。
2. 后端图编译：根据配置的多级编译级别，选择不同的编译模式，O0为最基础的原生构图与编译，O1在O0基础增加了自动算子融合功能，主要功能有图优化、图算融合、算子选择、执行序编排，其中图算融合为O1模式下独有功能。
3. 后端图执行：O0跟O1模式执行层面是一样的，均使用单算子方式调度执行，主要功能有多流并发、多级流水、HAL管理、内存管理。

## O0模式介绍

O0为基础的图编译执行模式，除必要影响功能的优化外，其他优化均关闭，使用原生的图结构进行编译和执行，方便调试调优，具备较好的编译性能。下面主要介绍后端图编译相关功能，后端图执行相关功能详见[运行时](https://www.mindspore.cn/docs/zh-CN/r2.6.0/features/runtime/memory_manager.html)。

### 图优化

O0模式的图优化较少，基础的优化主要为后端LazyInline和No-task node执行优化。

- **后端LazyInline**

  **LazyInline**：主要思想是将函数调用的开销推迟到实际需要调用的时候，这样可以减少编译时的开销，提高编译效率。LazyInline在图编译阶段是将相同的子图结构复用，不展开放在图中，避免图规模较大导致影响编译性能。

  ![jit_level_lazyinline](./images/multi_level_compilation/jit_level_lazyinline.png)

  **流水线（Pipeline）并行**：将神经网络中的算子切分成多个Stage，再把Stage映射到不同的设备上，使得不同设备去计算神经网络的不同部分。为了提升效率，流水线并行进一步将小批次(MiniBatch)切分成更细粒度的微批次(MicroBatch)，在微批次中采用流水线式的调度，从而达到提升效率的目的。

  **后端LazyInline**：由于Pipeline并行的MicroBatch切分会导致整个计算图扩张到MicroBatch的数量倍，从而导致模型规模巨大，编译性能时间较长（可能小时级别）。而这些Micro子图结构都是一样的，为了解决编译性能问题，LazyInline技术则非常契合，不过LazyInline带来的问题就是运行时无法采用最优的方式进行内存复用和流分配、无法做跨图的优化（内存优化、通信融合、算子融合等）等问题。为此，在图编译结束后，在图执行之前，将这些Micro子图做实际的节点Inline，以形成完整的全局整图，再通过图Inline后的内存优化、通信优化、冗余计算消除等方式，从而实现在编译性能、执行性能、执行内存方面都兼顾的目标。

- **No-task node执行优化**

  ![jit_level_no_task](./images/multi_level_compilation/jit_level_no_task.png)

  No-task node指的是Reshape、ExpandDims、Squeeze、Flatten、FlattenGrad、Reformat等诸类算子没有计算逻辑，不修改内存排布，仅修改shape、format等信息。在图编译结束后，将No-task node转换成ref node，输出跟输入同地址，执行过程中跳过kernel launch，从而达到执行性能优化目的。

### 算子选择

算子是深度学习框架中的基本执行单元，它们负责执行特定的计算任务，如矩阵乘法、卷积、池化等。算子选择需要综合考虑算子类型、数据类型、硬件平台和算子优化等因素，以选择最优的算子来实现深度学习任务。

MindSpore Ascend后端的算子类型有Aclnn kernel/Aclop kernel/Hccl kernel /Cpu kernel，算子选择流程如下图所示：

![jit_level_kernelselect](./images/multi_level_compilation/jit_level_kernelselect.png)

1. 算子类型：首先根据算子类型选择为计算算子还是通信算子。
2. 硬件平台：如果硬件上有对应算子，则优先选择硬件上的算子，否则选择CPU上的异构算子，例如shape相关的计算算子可能只适合在CPU上支持，没有对应的硬件算子。
3. 算子效率：Ascend上由于Aclnn算子较好的性能，因此计算类型算子如果有对应Aclnn kernel，则优先选择Aclnn kernel，否则就选择Aclop kernel。
4. 如果上述3步都未选择到算子，则为不支持的算子，算子选择失败退出。

### 执行序编排

![jit_level_exec_order](./images/multi_level_compilation/jit_level_exec_order.png)

不同图遍历算法产生的执行序在执行性能跟内存上会有较大的差异，如上图所示：

- **BFS得到的执行序**：kernel1-> kernel2-> kernel4-> kernel5-> kernel3-> kernel6，内存峰值为5G（kernel3执行后可以把kernel1和kernel2的释放掉，则轮到kernel6执行的时候则能复用，因此kernel6 不用额外申请多的内存）。
- **DFS得到的执行序**：kernel1-> kernel2-> kernel3-> kernel4-> kernel5-> kernel6，内存峰值为4G（kernel3执行后可以把kernel1和kernel2的释放掉，则轮到kernel4和kernel5执行的时候则能复用，因此kernel4和kernel5不用额外申请多的内存）。

执行序编排是在一定内存限制下求解最优算子并发的复杂性问题，不仅需要识别和利用计算图中的并发机会，以提升计算效率，还必须同时考虑多种限制条件，以确保系统的稳定性和高效性。

- 首先，优化模块需要解决求解最优算子并发的复杂性问题。由于计算图中的算子数量庞大且相互依赖，找到一个既能最大化并发又能保持计算图逻辑正确性的执行顺序是一个极具挑战性的任务。
- 其次，内存限制是执行序优化中不可忽视的关键因素。增大并发虽然可以提升计算效率，但往往会显著增加峰值内存需求，从而可能导致内存溢出（OOM）错误，尤其是在资源受限的环境中。因此，优化模块必须权衡并发与内存使用之间的关系，确保在提升并发的同时，不会超出系统的内存容量。
- MindSpore的执行序调整模块结合了基于规则和基于启发式策略的方式，提供bfs/dfs两种执行序编排算法[mindspore.jit(option={"exec_order":"bfs/dfs"})](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.jit.html#mindspore.jit)，以实现对计算图执行顺序的精细调整，从而在保证计算效率的同时，有效应对内存限制和系统稳定性等多重挑战。

## O1模式介绍

O1主要定位于在O0基础上实现通用、可泛化的AI编译优化，以支持大部分通用训练、推理场景的更好执行性能需求。

在当前阶段，O1主要支持了图算融合优化。其主要思路是：在静态图编译阶段，自动识别计算图中相邻的可融合节点，然后将其融合为更大粒度的可执行算子。通过图算融合，实现增加算子计算局部性、减少整体全局内存访存带宽开销等优化效果。通过对15+网络的实测验证，O1能够实现相比O0平均15%的性能加速。特别是对于访存密集型网络，O1优化效果更加显著。

### 图算融合

MindSpore等主流AI计算框架对用户提供的算子通常是从用户可理解、易使用角度进行定义。每个算子承载的计算量不等，计算复杂度也各不相同。但从硬件执行角度看，这种天然的、基于用户角度的算子计算量划分，并不高效，也无法充分发挥硬件资源计算能力。主要体现在：

1. 计算量过大、过复杂的算子，通常很难生成切分较好的高性能算子，从而降低设备利用率；
2. 计算量过小的算子，由于计算无法有效隐藏数据搬移开销，也可能会造成计算的空等时延，从而降低设备利用率；
3. 硬件Device通常为多核、众核结构，当算子shape较小或其他原因引起计算并行度不够时，可能会造成部分核的空闲，从而降低设备利用率。特别是基于专用处理器架构（Domain Specific Architecture，后文简称DSA）的芯片对这些因素更为敏感。如何最大化发挥硬件算力性能的同时使算子也能具备较好的易用性，一直以来是一个很大的挑战。

在AI框架设计方面，目前业界主流采用图层和算子层分层的实现方法。图层负责对计算图进行融合或重组，算子层负责将融合或重组后的算子编译为高性能的可执行算子。图层通常采用基于Tensor的High-Level IR的处理和优化，算子层则采用基于计算指令的Low-Level IR进行分析和优化。 这种人为分层处理显著增加了图、算两层进行协同优化的难度。

MindSpore在过去几年的技术实践中，采用了图算融合的技术来较好的解决了这个问题。NLP、推荐等不同类别的典型网络在使能图算融合后训练速度都有明显收益。主要原因之一就是这些网络中存在大量小算子组合，具有较多的融合优化机会。

#### 图算融合架构及整体流程

图算融合整体架构如下图所示。在图层主要思路是把复合算子打开，然后进行跨边界聚合和优化，最后进行Kernel算子拆分。主要步骤包括：

1. Composite Expansion：将复合算子展开为基本算子，并构成Composite子图，方便进行后续的跨边界优化和算子拆分；
2. Cross-OP Aggregation：将相邻的基本算子或Composite子图进行聚合，从而构成更大的聚合子图，方便进行后续的跨边界优化和算子拆分；
3. High-Level Optimization：在上面两步得到的聚合子图的基础上，我们可以进行大量的跨边界优化，如代数化简、公共子表达式提取（CSE）等；
4. Kernel Partition：基于计算特征以及融合算子性能，对聚合计算子图进行算子拆分。

优化后的计算图会以一个个子图的方式传给MindSpore AKG继续进行后端优化、生成目标代码。

![graphkernel](./images/graphkernel.png)

通过以上步骤，我们可以获得两方面性能收益：

1. 不同算子之间的跨边界性能优化收益；
2. 通过对整个计算图进行重组拆分，得到最优粒度的融合算子。

#### 融合算子加速优化（MindSpore AKG）

前文提到，在HPC、深度神经网络训练等场景中，图算融合优化可带来成倍的性能提升。但随着图算融合能力的不断增强，融合算子的开发成为了继续提升图算融合能力的瓶颈点。

融合算子的自动生成技术可以解决基于DSA开发融合算子编程门槛较高的问题，让程序员在算子开发过程中能够聚焦于算子的实现逻辑，无需关注后端优化，极大提高其开发效率。尤其对于后端硬件架构复杂以及存在复杂算子和融合算子的场景，算子自动生成技术更加关键。

因此，**MindSpore AKG基于多面体编译技术（Polyhedral Model），对融合算子的加速优化与自动生成**，能够帮助MindSpore的图算融合模块优化后的融合算子在**异构硬件平台**（GPU/Ascend）上自动生成高性能的kernel，提升MindSpore的训练性能。

架构及整体流程如下：

![graphkernel_akg_overview](./images/graphkernel_akg_overview.png)

MindSpore AKG的整体框架如上图所示：

- IR规范化
    - MindSpore AKG的输入为MindSpore图算融合模块优化后的融合子图，通过TVM的Compute / IR Builder / Hybrid 等多种描述方式对子图中的算子进行表达。然后DSL会被转换为 Halide IR（[Halide](https://halide-lang.org/)，是常见的用于开发高性能图像处理和Array计算的语言，可作为中间表达解耦算法和优化）并进行 IR 规范化；
    - 完成初步简化和优化后，Halide IR会被转化为Poly模块所需的调度树；
- Poly模块调度优化
    - 利用Polyhedral技术中的Pluto调度算法，实现循环的自动融合、自动重排等变换，为融合算子自动生成满足并行性、数据局部性的初始调度；
    - 为快速适配不同硬件后端，Poly模块内的优化pass会分为硬件无关的通用优化与硬件相关的特定优化，编译时按照硬件特征拼接组合，实现异构硬件后端的快速适配。自动切分、自动映射以及自动内存提升等pass会根据不同硬件的架构性质给出不同的优化方式；
- 后端优化
    - 为了进一步提升算子的性能，我们针对不同硬件后端开发了相应的优化pass，如Ascend后端中实现数据对齐、指令映射，GPU后端中实现向量化存取，插入同步指令等，最终生成相应平台代码。

### 其它图优化技术

除了图算融合之外，在后续版本中，O1可能会逐步扩展增加一些其它图优化技术。比如：

1. KernelPacket: 用于在动态shape场景对shape计算进行自动融合和优化；
2. 通算融合：将通信算子与计算算子进行融合。
