# 数据处理性能调试

`Linux` `Ascend` `GPU` `CPU` `中级` `高级`

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [数据处理性能调试](#数据处理性能调试)
    - [概述](#概述)
    - [脚本撰写](#脚本撰写)
    - [操作系统的影响](#操作系统的影响)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced_use/data_processing_acceleration.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

数据处理的性能涉及到多方面的因素，包括脚本撰写、操作系统等。

## 脚本撰写

数据处理的脚本大致分为以下几个模块：

![dataset_pipeline](./images/dataset_pipeline.png)

- 数据加载

    数据加载的方式有三种：
    1. 内置高性能的数据加载类算子，如CIFAR数据集、MNIST数据集等；
    2. 将数据集转换成MindRecord，使用`MindDataset`算子进行加载；
    3. 用户自定义数据集——`GeneratorDataset`。

    > 优先使用内置的数据加载类算子以及`MindDataset`，如果无法满足用户需求，则在撰写用户自定数据集加载时，需要关注本身数据集加载的性能优化。

- 数据混洗

    ![shuffle](./images/shuffle.png)

    > `shuffle`操作主要用来将数据混洗，设定的`buffer_size`越大，混洗程度越大，但时间、计算资源消耗会大。因此该算子我们不建议使用，现在数据加载类算子中可以支持`shuffle`的功能。

- 数据增强

    数据增强的方式有三种：
    1. C算子的数据增强(C++)；
    2. Python算子的数据增强(Pillow)；
    3. 用户自定义的数据增强(Python function)。

    > 优先使用C算子的数据增强。根据用户自定义的数据增强的算子类型进行多线程还是多进程模式的选择，计算密集型使用多进程，IO密集型使用多线程。

- `batch` & `repeat`

    `batch`和`repeat`一般不会成为性能瓶颈。

## 操作系统的影响

由于数据处理是在host端进行，那么机器或者操作系统本身的一些配置会对数据处理存在影响，主要有存储、NUMA架构、CPU（计算资源）几个方面。

1. 存储

    当数据集较大时，我们推荐使用固态硬盘对数据进行存储，能够减少存储I/O对于数据处理的影响。

    > 一般地，当数据集被加载之后，就会缓存在操作系统的page cache中，在一定程度上降低了存储开销，加快了后续epoch的数据读取。

2. NUMA架构

    非一致性内存架构(Non-uniform Memory Architecture)是为了解决传统的对称多处理(Symmetric Multi-processor)系统中的可扩展性问题而诞生的。NUMA系统拥有多条内存总线，于是将几个处理器通过内存总线与一块内存相连构成一个组，这样整个庞大的系统就可以被分为若干个组，这个组的概念在NUMA系统中被称为节点(node)。处于该节点中的内存被称为本地内存(local memory)，处于其他节点中的内存对于该组而言被称为外部内存(foreign memory)。因此每个节点访问本地内存和访问其他节点的外部内存的延迟是不相同的，在数据处理的过程中需要尽可能避免这一情况的发生。一般我们可以使用以下命令进行进程与node节点的绑定：

    ```shell
    numactl --cpubind=0 --membind=0 python train.py
    ```

    上述例子表示将此次运行的`train.py`的进程绑定到`numa node` 0上。

3. CPU（计算资源）

    CPU对于数据处理的影响主要是计算资源的分配和CPU频率的设置两个方面。

    - 计算资源的分配

        当我们进行分布式训练时，一台设备机器上会启动多个训练进程，而这些训练进程会通过操作系统本身的策略进行计算资源的分配与抢占，当进程较多时，可能会由于计算资源的竞争而导致数据处理性能的下降，因此这时需要进行人工分配计算资源，避免各个进程的计算资源竞争。

        ```shell
        numactl --cpubind=0 python train.py
        or
        taskset -c 0-15 python train.py
        ```

        > `numactl`的方式较为粗粒度，直接指定`numa node id`，而`taskset`的方式是细粒度的，它能够直接指定`numa node`上的`cpu core`，其中0-15表示的`core id`从0到15。

    - CPU频率设置

        要想充分发挥host端CPU的最大算力，CPU频率的设置至关重要。一般地，linux内核支持调节CPU主频，降低功耗，已到达节能的效果。通过选择系统空闲状态不同的电源管理策略，可以实现不同程度降低服务器功耗。但是，更低的功耗策略意味着CPU唤醒更慢对性能影响更大。因此如果发现CPU模式为conservative或者powersave，可以使用cpupower设置CPU Performance模式，对数据处理的性能提升有非常大的效果。

        ```shell
        cpupower frequency-set -g performance
        ```
