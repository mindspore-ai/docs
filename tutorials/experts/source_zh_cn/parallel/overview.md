# 分布式并行总览

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/overview.md)

在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。此外，分布式并行对大模型训练和推理有着重要的意义，它为处理大规模数据和复杂模型提供了强大的计算能力和性能优势。

要实现分布式并行训练和推理，您可以参考以下指引：

## 启动方式

MindSpore目前支持三种启动方式：

- **动态组网**：通过MindSpore内部动态组网模块启动，不依赖外部配置或者模块，支持Ascend/GPU/CPU。
- **mpirun**：通过多进程通信库OpenMPI启动，支持Ascend/GPU。
- **rank table**：配置rank_table表后，通过脚本启动和卡数对应的进程，支持Ascend。

详细可参考[启动方式](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/startup_method.html)章节。

## 并行模式

目前MindSpore可以采取下述的几种并行模式，您可以按需求选择：

- **数据并行模式**：数据并行模式下，数据集可以在样本维度拆分并下发到不同的卡上。如果您的数据集较大，而模型参数规模能在单卡运算，您可以选择这种并行模型。参考[数据并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/data_parallel.html)教程了解更多信息。
- **自动并行模式**：融合了数据并行、算子级模型并行的分布式并行模式，可以自动建立代价模型，找到训练时间较短的并行策略，为用户选择合适的并行模式。如果您的数据集和模型参数规模都较大，且希望自动配置并行策略，您可以选择这种并行模型。参考[自动并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/auto_parallel.html)教程了解更多信息。
- **半自动并行模式**：相较于自动并行，该模式需要用户对算子手动配置切分策略实现并行。如果您数据集和模型参数规模都较大，且您对模型的结构比较熟悉，知道哪些“关键算子”容易成为计算瓶颈，为“关键算子”配置合适的切分策略可以获得更好的性能，您可以选择这种并行模式。此外该模式还可以手动配置优化器并行和流水线并行。参考[半自动并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/semi_auto_parallel.html)教程了解更多信息。
- **手动并行模式**：在手动并行模式下，您可以基于通信原语例如`AllReduce`、`AllGather`、`Broadcast`等通信算子进行数据传输，手动实现分布式系统下模型的并行通信。您可以参考[手动并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/manual_parallel.html)教程了解更多信息。
- **参数服务器模式**：相比于同步的训练方法，参数服务器具有更好的灵活性、可拓展性以及节点容灾能力。您可以参考[参数服务器](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html)模式教程了解更多信息。

## 保存和加载模型

模型的保存可以分为合并保存和非合并保存，可以通过`mindspore.save_checkpoint`或者`mindspore.train.CheckpointConfig`中的`integrated_save`参数选择是否合并保存。合并保存模式下模型参数会自动聚合保存到模型文件中，而非合并保存模式下每张卡保存各自卡上的参数切片。关于各并行模式下的模型保存可以参考[模型保存](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_saving.html)教程。

模型的加载可以分为完整加载和切片加载。若保存的是完整参数的模型文件，则可以直接通过`load_checkpoint`接口加载模型文件。若保存的是多卡下的参数切片文件，则需要考虑加载后的分布式切分策略或集群规模是否有变化。如果分布式切分策略或集群规模不变，则可以通过`load_distributed_checkpoint`接口加载各卡对应的参数切片文件，可以参考[模型加载](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_loading.html)教程。

若保存和加载的分布式切分策略或集群卡数改变的情况下，则需要对分布式下的Checkpoint文件进行转换以适配新的分布式切分策略或集群卡数。您可以参考[模型转换](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_transformation.html)了解更多信息。

## 故障恢复

在分布式并行训练过程中，可能会遇到计算节点的故障或通信中断等问题。MindSpore提供了两种恢复方式以保证训练的稳定性和连续性：

- **根据完整Checkpoint恢复**：在保存Checkpoint文件前，通过AllGather算子汇聚模型的完整参数，每张卡均保存了完整的模型参数文件，可以直接加载恢复。多副本提高了模型的容错性，但是对于大模型来说，汇聚的过程会导致各种资源开销过大。
- **根据参数切分的冗余信息恢复**：在大模型训练中，根据数据并行的维度所划分的设备，他们的模型参数是相同的。根据这个原理，可以利用这些冗余的参数信息作为备份，在一个节点故障时，利用相同参数的另一节点就可以恢复故障的节点。详细可参考[故障恢复](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/fault_recover.html)教程。

## 优化方法

如果对性能、吞吐量或规模有要求，或者不知道如何选择并行策略，可以考虑以下优化技术：

- **并行策略优化**：
    - **策略选择**：根据您的模型规模和数据量大小，您可以参考[策略选择]()教程来选择不同的并行策略，以提高训练效率和资源利用率。
    - **切分技巧**：切分技巧也是实现高效并行计算的关键，在[切分技巧]()教程中，您可以通过具体案例了解到如何应用各种切分技巧来提升效率。
    - **多副本**：在现有的单副本模式下，某些底层算子在进行通信的时候，无法同时进行计算，从而导致资源浪费。多副本模式通过对数据按照batchsize维度进行切分为多个副本，可以使一个副本在通信时，另一副本进行计算操作，提升了资源利用率，详细可参考[多副本]()教程。
- **内存优化**：
    - **重计算**：重计算通过不保存正向算子的计算结果，以节省内存空间，在计算反向算子时，需要用到正向结果再重新计算正向算子。详细可参考[重计算](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/recompute.html)教程。
    - **数据集切分**：数据集单个数据过大的时候，可以对数据进行切分，进行分布式训练。数据集切分配合模型并行是有效降低显存占用的方式。详细可参考[数据集切分](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dataset_slice.html)教程。
    - **Host&Device异构**：在遇到参数量超过Device内存上限的时候，可以把一些内存占用量大且计算量少的算子放在Host端，这样能同时利用Host端内存大，Device端计算快的特性，提升了设备的利用率。详细可参考[Host&Device异构](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/host_device_training.html)教程。
    - **异构存储**：大模型目前受限显存大小，难以在单卡上训练。大规模分布式集群训练中，在通信代价越来越大的情况下，提升单机的显存，减少通信，也能提升训练性能。异构存储可以将暂时不需要用到的参数或中间结果拷贝到Host端内存或者硬盘，在需要时再恢复至Device端。详细可参考[异构存储](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/memory_offload.html)教程。
- **通信优化**：
    - **通信融合**：通信融合可以将相同源节点和目标节点的通信算子合并到一次通信过程，避免多次通信带来额外开销。详细可参考[通信融合](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/comm_fusion.html)。
    - **通信子图提取与复用**：通过对通信算子提取通信子图，替换原本的通信算子，可以减少通信耗时，同时减少模型编译时间。详细可参考[通信子图提取与复用](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/comm_subgraph.html)。
