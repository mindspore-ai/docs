# MindPandas执行模式介绍及配置说明

<a href=“https://gitee.com/mindspore/docs/blob/master/docs/mindpandas/docs/source_zh_cn/mindpandas_performance.md” target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

本文会介绍下MindPandas的分布式并行模式的使用，以及MindPandas的性能优势。

## MindData执行原理

MindData通过分布式并行化执行的方式实现性能的显著提升。其原理是首先对原始数据进行分片，再将API转化为通用计算范式（map、reduce、injective_map等），之后由后端并行化执行。当前MindPandas后端有两种执行模式，分别是多线程和多进程，其中多进程模式支持单机多进程与多机多进程。

### 数据分片

将原始数据分片是并行化执行的基础，如下图所示，将`pandas.DataFrame`转换为`minddata.DataFrame`过程，根据预设的`partition_shape`将原始数据分割为指定数量的`partition`，`partition`将作为后续并行化执行的基本单位。

![partition.png](images/partition.png)

### 多线程模式

多线程模式基于python多线程实现，将每个分片以及对应的计算任务作为一个线程提交到python线程池，由线程池管理并发执行。

![multithread.png](images/multithread.png)

虽然python的多线程存在GIL限制，导致多线程无法有效利用多核，但较小的数据量或IO密集型的任务，使用多线程后端仍能带来显著的性能提升。

### 多进程模式

多进程模式支持单机和多机，且多进程模式不受python的GIL限制，可以做到真正的并行计算。多进程模式与多线程模式整体原理类似，不同的是在对原始数据进行切片后，会将分片存入分布式计算引擎的共享内存中，`minddata.DataFrame`中存放的则是分片所对应的`object reference`。

当需要进行计算时，会将计算函数也存入分布式计算引擎的共享内存中， 之后将计算函数对应的`object reference`与分片对应的`object reference`作为一个任务提交到分布式计算引擎，所有任务会由分布式计算引擎统一调度，以多进程的形式异步并行执行。

#### 单机多进程

![multiprocess1.png](images/multiprocess1.png)

多进程模式可以充分利用多核，从而实现数倍到数十倍不等的性能提升。因此多进程模式能够非常高效的处理海量数据，不过由于进程创建、调度等开销，在处理小数据量时可能效果会受影响。

#### 多机多进程

![multiprocess2.png](images/multiprocess2.png)

多机多进程模式可以将多台服务器组成集群，充分利用多台服务器的资源完成计算任务，从而突破单机的资源限制。

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

MindPandas内置的分布式执行引擎，是一个分布式对象框架，它使用了和传统分布式计算系统不一样的架构。

安装MindPandas时，内置的分布式执行引擎也已经同步安装完成，可以在控制台输入yrctl访问。

```shell
yrctl
```

要使用分布式执行引擎，我们需要先在主节点通过命令行启动服务，其中address为主节点的IP地址，如下所示：

```shell
yrctl start --master --address <address>
```

yrctl start 命令参数如下：

--master 主节点标注，非必选项。

--address 主节点的IP地址，为必选项。

--password Etcd密码，必选项，不大于64个字符。

--cpu 数据系统使用的cpu单位，单位s是1/1000核，非必选项，默认系统资源的30%。

--mem 数据系统使用的内存，单位是MB，非必选项，默认系统内存的1/3。

--datamem 数据系统使用的内存，单位是MB，非必选项，默认系统内存的1/6。

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

若当前节点为主节点，则需要设置此参数。

### 多机多进程模式使用

通常在实践中，我们有大量的数据需要处理，需要超越单台机器的能力。尽管MindPandas在本地模式已经可以正常工作并表现良好，但是我们还可以选择集群环境让计算变得更快。用户无需考虑存在多少工作人员或如何分配和分区他们的数据，MindPandas会无缝且透明地为您处理所有的这些问题。

集群由运行在不同机器上的多个节点组成，每个集群有一个“头节点”和多个“工作节点”，所有节点必须在同一个本地网络上，通过在每台机器上启动一个主进程来创建一个集群，命令如下：

启动master结点

```shell
yrctl start --master --address=<address> --password=<password>
```

其中address为主节点的IP地址。

启动工作节点

```shell
yrctl start --address=<address> --password=<password>
```

其中address为主节点的IP地址，在集群上运行分布式程序，需要在与其中一个节点相同的机器上执行程序。

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

### 自适应并发模式

MindPandas提供了一种自动切换并发模式的自适应功能，即MindPandas通过检测录入csv文件的大小或者输入DataFrame/Series在CPU内存的占用情况，自主更换其并发模式以提升脚本的运算性能。

#### 启动自适应并发模式

MindPandas通过config的```get_adaptive_concurrency```和```set_adaptive_concurrency```对自适应并发模式功能的使用进行获取和设置。

MindPandas的自适应并发模式功能默认设置为关闭状态，即

```python
import mindpandas as mpd
print(mpd.config.get_adaptive_concurrency)
##The print result is False
```

如用户想要启动自适应并发模式功能，可以通过set_adaptive_concurrency接口轻松打开该功能，如下：

```python
import mindpandas as mpd
mpd.config.set_adaptive_concurrency(True)
```

#### 工作原理

当自适应并发模式功能被启动，MindPandas后端通过检测csv文件大小来自动切换并发模式，即自主选择使用多线程模式并发模式或者多进程并发模式，其切换标准如下：

- 针对.csv格式文件，小于18MB的csv文件采用多线程并发模式，其他文件采用多进程并发模式。

- 针对以pandas.DataFrame初始化的mpd.DataFrame，CPU内存使用小于1GB的将采用多线程并发模式，其他则采用多进程并发模式。

- 针对以numpy.array初始化的mpd.DataFrame，CPU内存使用小于1GB的将采用多线程并发模式，其他则采用多进程并发模式。

#### 注意事项

- 自适应并发模式被启动后，并行模式和分区形状均由MindPandas后端自主调整，所以用户无法再使用set_concurrency_mode对并发模式进行修改。

- set_adaptive_concurrency(True)应在脚本开头调用，以确保每个DataFrame的并发操作都已按照阈值大小进行设置。

- 在设置set_adaptive_concurrency(True)后，除非end2end脚本已完整运行结束，不建议用户将自适应并发模式切换回False。

#### 使用限制

- 自适应并发模式功能目前不支持来自merge、concat或join等操作所创建的DataFrame。

- 自适应并发模式功能无法更改在启动该功能前初始化或读入的DataFrame/Series的并发模式。

- 自适应并发模式功能目前使用特定的分片形状，即多线程模式采用(2, 2)的分片，多进程模式采用(16, 16)的分片。

- 除read_csv之外的其他I/O操作，例如read_feather，目前不支持自适应并发模式功能。
