# 日志

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/function/logs.md)

## 日志保存

### 概述

MindSpore TransFormers 会将模型的训练配置、训练步数、Loss、吞吐率等信息写入日志中，开发者可以自行指定日志存储的路径。

### 训练日志的目录结构

在训练过程中，MindSpore Transformers 默认会在输出目录（默认为 `./output` ）中生成训练日志目录： `./log` 。

而当使用 `ms_run` 方式启动训练任务时，将会默认同时在输出目录下额外生成日志目录： `./msrun_log` 。

| 文件夹        | 描述                                                                                                                                            |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| log        | 以 `rank_{i}` 文件夹来划分保存每一张卡的日志信息。（ `i` 对应为训练任务所用的 NPU 卡号）<br>每一个 `rank_{i}` 文件夹底下将包括 `info.log` 和 `error.log` 来分别记录训练时输出的 INFO 级别和 ERROR 级别的信息。 |
| msrun_log  | 以 `worker_{i}.log` 来记录每一张卡的训练日志（包括报错信息）， `scheduler.log` 则记录了 msrun 的启动信息。<br>一般更常通过此文件夹查看训练日志信息。                                             |

以一个使用 `msrun` 当时启动的 8 卡任务为例，具体日志结构如下所示：

```text
output
    ├── log
        ├── rank_0
            ├── info.log    # 记录 0 号卡的训练信息
            └── error.log   # 记录 0 号卡的报错信息
        ├── ...
        └── rank_7
            ├── info.log    # 记录 8 号卡的训练信息
            └── error.log   # 记录 8 号卡的报错信息
    └── msrun_log
        ├── scheduler.log   # 记录各张卡之间的通信信息
        ├── worker_0.log    # 记录 0 号卡的训练及信息
        ├── ...
        └── worker_7.log    # 记录 8 号卡的训练及信息
```

### 配置与使用

MindSpore TransFormer 默认会在训练的 yaml 文件中指定文件输出路径为 `./output` 。如果在 `mindformers` 路径下启动训练任务，则训练产生的日志输出将默认保存在 `mindformers/output` 下。

#### YAML 参数配置

如果需要重新指定输出的日志文件夹，可以在 yaml 中修改配置。

以 [`DeepSeek-V3` 预训练 yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L2) 为例，可做如下配置：

```yaml
output_dir: './output' # path to save logs/checkpoint/strategy
```

#### 单卡任务指定输出目录

除了 yaml 文件配置来指定，MindSpore TransFormer 还支持在 [run_mindformer 一键启动脚本](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/start_tasks.html?highlight=%E6%97%A5%E5%BF%97#run-mindformer%E4%B8%80%E9%94%AE%E5%90%AF%E5%8A%A8%E8%84%9A%E6%9C%AC) 中，使用 `--output_dir` 启动命令对日志输出路径做指定。

> 如果在这里配置了输出路径，将会覆盖 yaml 文件中的配置！

#### 分布式任务指定输出目录

如果模型训练需要用到多台服务器，使用[分布式任务拉起脚本 msrun_launcher.sh](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/start_tasks.html?highlight=%E6%97%A5%E5%BF%97#%E5%88%86%E5%B8%83%E5%BC%8F%E4%BB%BB%E5%8A%A1%E6%8B%89%E8%B5%B7%E8%84%9A%E6%9C%AC) 来启动分布式训练任务的话，

在设置了共享存储的情况下，还可以在启动脚本中指定入参 `LOG_DIR` 来指定 Worker 以及 Scheduler 的日志输出路径，将所有机器节点的日志都输出到一个路径下，方便统一观察。