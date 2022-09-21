# MindPandas后端执行模式配置及性能介绍

<a href=“https://gitee.com/mindspore/docs/blob/r1.9/docs/mindpandas/docs/source_zh_cn/mindpandas_performance.md” target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

本文会介绍下MindPandas的分布式并行模式的使用，以及MindPandas的性能优势。

## MindPandas的多模式后端配置

MindPandas的性能优势体现在使用了分布式后端执行模式，运行脚本之前需要进行配置，有如下三个场景。

### 单机多线程模式使用

在MindPandas中我们默认使用多线程模式。多线程模式通过Python多线程实现，并且对较大的DataFrame自动进行了切片存储。用户只需通过导入MindPandas而无需重新编写脚本便可实现原脚本的多线程加速。
Python脚本中处理如下：

```Python
import mindpandas as pd
pd.set_concurrency_mode('multithread') # MindPandas will use multithread as backend

df = pd.read_csv('data.csv')
df_mean = df.mean()
```

### 单机多进程模式使用

MindPandas内置的分布式执行引擎，是一个分布式对象框架，它使用了和传统分布式计算系统不一样的架构和对分布式计算的抽象方式。

安装MindPandas时，内置的分布式执行引擎也已经同步安装完成，可以在控制台输入yrctl访问。

```shell
yrctl
```

要使用分布式执行引擎，我们需要先在master节点通过命令行启动服务，如下所示：

```shell
yrctl start --master --address <address>
```

回显如下：

```shell
Succeeded to deploy the function func!
Succeeded to start!
```

如需查看yrctl start的更多的参数使用说明，可以使用--help参数。

在数据处理脚本中，设置使用yr后端，如当前主机地址是192.168.1.1，安装数据系统服务地址也是本机，Python脚本如下设置。

```Python
import mindpandas as pd
pd.set_concurrency_mode("yr", server_address="192.168.1.1", ds_address="192.168.1.1")
```

脚本执行完后，停止分布式执行引擎，使用如下命令：

```shell
yrctl stop --master
```

若当前节点为master节点，则需要设置此参数。

### 多机多进程模式使用

通常在实践中，我们有大量的数据需要处理，需要超越单台机器的能力。尽管MindPandas在本地模式已经可以正常工作并表现良好，但是我们还可以选择集群环境让计算变得更快。用户无需考虑存在多少工作人员或如何分配和分区他们的数据，MindPandas会无缝且透明地为您处理所有的这些问题。

集群由运行在不同机器上的多个节点组成，每个集群有一个“头节点”和多个“工作节点”，所有节点必须在同一个本地网络上，通过在每台机器上启动一个主进程来创建一个集群，命令如下：

启动master结点

```shell
yrctl start --master --address=<address> --password=<password>
```

启动工作节点

```shell
yrctl start --address=<address> --password=<password>
```

在集群上运行分布式程序，需要在与其中一个节点相同的机器上执行程序。
在您的程序/脚本中，您必须调用ray.init并将address参数添加到ray.init（如ray.init(address=...)）。这会导致您的脚本连接到集群上现有的 Ray 运行时。例如：
在数据处理脚本中，设置使用yr后端，如当前主机地址是192.168.1.1，安装数据系统服务地址也是本机，python脚本如下设置。

```Python
import mindpandas as pd
pd.set_concurrency_mode("yr", server_address="192.168.1.1", ds_address="192.168.1.1")
```

运行结束后删除集群使用如下命令，其中在主节点机器执行：

```shell
yrctl stop --master
```

在从节点机器执行：

```shell
yrctl stop
```

### MindPandas 中的自适应并发

Mindpandas 提供了使 DataFrames 的并发模式适应其大小和内存使用情况的功能。支持自适应并发的主要操作是read_csv。自适应并发将按如下方式工作：

> - read_csv 将对小于 18 MB 的 csv 文件使用多线程运算符，否则使用多进程运算符
> - 从 pandas DataFrame 初始化的 DataFrame 将对 CPU 使用率小于 1 GB 的 DataFrame 使用多线程运算符，否则使用多进程运算符
> - 从 numpy 数组初始化的 DataFrame 会将数组转换为 pandas DataFrame，并对 CPU 使用率小于 1 GB 的 DataFrame 使用多线程运算符，否则使用多进程运算符

#### 设置自适应并发

```Python
import mindpandas as mpd
mpd.set_adaptive_concurrency(True)
```

> - read_csv 现在将根据上述 18 MB 阈值设置 DataFrame 的并发模式

#### 关于自适应并发的重要说明

一旦将自适应并发设置为 True，就会抛出对 mpd.set_concurrency_mode 的调用。用户不应调用 mpd.set_concurrency_mode 或 mpd.set_partition_shape。这是因为自适应并发会自动为每个 DataFrame 选择并发运算符和分区形状(Partition Shape)，而不是尊重用户的设置。此外，mpd.set_adaptive_concurrency(True) 应在脚本开头调用，以确保每个 DataFrame 的并发操作都已按照阈值大小进行设置。

```Python
mpd.set_adaptive_concurrency(True)

# Any of the following will throw
mpd.set_concurrency_mode('yr')
mpd.set_concurrency_mode('multithread')
```

#### 关闭自适应并发

```Python
mpd.set_adaptive_concurrency(False)
```

不建议在设置为 True 后将自适应并发设置为 False。默认情况下，自适应并发设置为 False，设置为 True 后才会生效。

#### 自适应并发的限制

自适应并发有以下限制：

> - 自适应并发当前无法设置从数据帧扩展或缩减操作（如 df.merge、mpd.concat 或 df.join）创建的数据帧的并发模式。
> - 自适应并发无法更改在自适应并发设置为 True 之前初始化或读入的 DataFrame 的并发模式。
> - 基于实验数据，自适应并发将多线程模式与 (2,2) 分区形状和射线模式与 (16,16) 分区形状耦合，并且不为每种模式采用一定范围的分区形状。
> - 除了 read_csv 之外的其他 I/O 操作，例如 read_feather，当面不支持使用自适应并发。

## MindPandas性能介绍

样例：使用replace API 将DataFrame中的0替换为1

```Python
import mindpandas as mpd
import pandas as pd
import numpy as np
import time

ms_pd.set_adaptive_concurrency(False)
ms_pd.set_partition_shape((16, 2))

#############################################
### For the purpose of timing comparisons ###
#############################################
frame_data = np.random.randint(0, 100, size=(2**10, 2**8))
mdf = mpd.DataFrame(frame_data)
start = time.time()
max_data = mdf.replace(0,1)
end = time.time()
mtime = end - start

df = pd.DataFrame(frame_data)
start = time.time()
max_data = df.replace(0,1)
end = time.time()
ptime = end - start

print(ptime/mtime)
#############################################

```

MindPandas相较于原生Pandas的API快10倍，花费更少时间创建DataFrame，原生Pandas花费近1分钟的时间。
