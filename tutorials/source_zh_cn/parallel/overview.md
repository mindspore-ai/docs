# 分布式并行概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/parallel/overview.md)

在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。此外，分布式并行对大模型训练和推理有着重要的意义，它为处理大规模数据和复杂模型提供了强大的计算能力和性能优势。

要实现分布式并行训练和推理，您可以参考以下指引：

## 分布式并行启动方式

MindSpore目前支持四种启动方式：

- **msrun**：是动态组网的封装，允许用户使用单命令行指令在各节点拉起分布式任务，安装MindSpore后即可使用，不依赖外部配置或者模块，支持Ascend/GPU/CPU。
- **动态组网**：通过MindSpore内部动态组网模块启动，不依赖外部配置或者模块，支持Ascend/GPU/CPU。
- **mpirun**：通过多进程通信库OpenMPI启动，支持Ascend/GPU。
- **rank table**：配置rank_table表后，通过脚本启动和卡数对应的进程，支持Ascend。

详细可参考[分布式并行启动方式](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/startup_method.html)章节。

## 数据并行

数据并行是最常用的并行训练方式，用于加速模型训练和处理大规模数据集。在数据并行模式下，训练数据被划分成多份，然后将每份数据分配到不同的计算节点上，例如多卡或者多台设备。每个节点独立地处理自己的数据子集，并使用相同的模型进行前向传播和反向传播，最终对所有节点的梯度进行同步后，进行模型参数更新。

详细可参考[数据并行](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/data_parallel.html)章节。

## 算子级并行

随着深度学习的发展，网络模型正变得越来越大，如NLP领域已出现万亿级参数量的模型，模型容量远超单个设备的内存容量，导致单卡或数据并行均无法进行训练。算子级并行将网络模型中每个算子涉及到的张量进行切分，并分配到多个设备上，降低单个设备的内存消耗，从而使大模型的训练成为可能。

MindSpore提供两种粒度的算子级并行能力：算子级并行和高阶算子级并行。算子级并行通过简单切分策略描述张量维度分布，满足大多数场景需求。高阶算子级并行通过开放设备排布描述，支持复杂切分场景。

详细可参考[算子级并行](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/operator_parallel.html)章节。

## 优化器并行

在进行数据并行训练时，模型的参数更新部分在各卡间存在冗余计算，优化器并行通过将优化器的计算量分散到数据并行维度的卡上，在大规模网络上（比如Bert、GPT）可以有效减少内存消耗并提升网络性能。

详细可参考[优化器并行](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/optimizer_parallel.html)章节。

## 流水线并行

近年来，神经网络的规模几乎是呈指数型增长。受单卡内存的限制，训练这些大模型用到的设备数量也在不断增加。受server间通信带宽低的影响，传统数据并行叠加模型并行的这种混合并行模式的性能表现欠佳，需要引入流水线并行。流水线并行能够将模型在空间上按阶段（Stage）进行切分，每个Stage只需执行网络的一部分，大大节省了内存开销，同时缩小了通信域，缩短了通信时间。MindSpore能够根据用户的配置，将单机模型自动地转换成流水线并行模式去执行。

详细可参考[流水线并行](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/pipeline_parallel.html)章节。

## 并行优化策略

如果对性能、吞吐量或规模有要求，或者不知道如何选择并行策略，可以考虑以下优化技术：

- **并行策略优化**：

    - **策略选择**：根据您的模型规模和数据量大小，您可以参考[策略选择](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/strategy_select.html)教程来选择不同的并行策略，以提高训练效率和资源利用率。
    - **切分技巧**：切分技巧也是实现高效并行计算的关键，在[切分技巧](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/split_technique.html)教程中，您可以通过具体案例了解到如何应用各种切分技巧来提升效率。
    - **多副本并行**：在现有的单副本模式下，某些底层算子在进行通信的时候，无法同时进行计算，从而导致资源浪费。多副本并行通过对数据按照Batch Size维度进行切分为多个副本，可以使一个副本在通信时，另一副本进行计算操作，提升了资源利用率，详细可参考[多副本并行](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/multiple_copy.html)教程。
    - **高维张量并行**：高维张量并行是指对于模型并行中的MatMul计算中的激活、权重张量进行多维度切分，通过优化切分策略降低通信量，提高训练效率，详细可参考[高维张量并行](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/high_dimension_tensor_parallel.html)教程。
- **内存优化**：

    - **梯度累加**：梯度累加通过在多个MicroBatch上计算梯度并将它们累加起来，然后一次性应用这个累加梯度来更新神经网络的参数。通过这种方法少量设备也能训练大Batch Size，有效减低内存峰值，详细可参考[梯度累加](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/distributed_gradient_accumulation.html)教程。
    - **重计算**：重计算通过不保存某些正向算子的计算结果，以节省内存空间，在计算反向算子时，需要用到正向结果再重新计算正向算子。详细可参考[重计算](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/recompute.html)教程。
    - **数据集切分**：数据集单个数据过大的时候，可以对数据进行切分，进行分布式训练。数据集切分配合模型并行是有效降低显存占用的方式。详细可参考[数据集切分](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/dataset_slice.html)教程。
    - **Host&Device异构**：在遇到参数量超过Device内存上限的时候，可以把一些内存占用量大且计算量少的算子放在Host端，这样能同时利用Host端内存大，Device端计算快的特性，提升了设备的利用率。详细可参考[Host&Device异构](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/host_device_training.html)教程。
- **通信优化**：

    - **通信融合**：通信融合可以将相同源节点和目标节点的通信算子合并到一次通信过程，避免多次通信带来额外开销。详细可参考[通信融合](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/comm_fusion.html)。

## 分布式高阶配置案例

- **基于双递归搜索的多维混合并行案例**：基于双递归搜索的多维混合并行是指用户可以配置重计算、优化器并行、流水线并行等优化方法，在用户配置的基础上，通过双递归策略搜索算法进行算子级策略自动搜索，进而生成最优的并行策略。详细可参考[基于双递归搜索的多维混合并行案例](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/multiple_mixed.html)。
- **在K8S集群上进行分布式训练**：MindSpore Operator是遵循Kubernetes的Operator模式（基于CRD-Custom Resource Definition功能），实现的在Kubernetes上进行分布式训练的插件。其中，MindSpore Operator在CRD中定义了Scheduler、PS、Worker三种角色，用户只需通过简单的YAML文件配置，就可以轻松地在K8S上进行分布式训练。MindSpore Operator的代码仓详见：[ms-operator](https://gitee.com/mindspore/ms-operator/)。详细可参考[在K8S集群上进行分布式训练](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/ms_operator.html)。
