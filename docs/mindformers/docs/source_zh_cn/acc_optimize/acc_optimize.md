# 大模型精度调优指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/acc_optimize/acc_optimize.md)

## 精度问题概述和场景

### 描述

随着昇腾AI处理器（以下简称为NPU）在深度学习中的广泛应用，基于昇腾NPU原生开发的MindSpore框架也展现出了更好的性能优势。在大规模集群训练过程中，性能的提升将极大节省用户进行大模型开发的成本。因此，越来越多的用户也逐渐将原本训练模型迁移至MindSpore中。然而，由于硬件以及框架使用上的差异，用户在完成模型迁移后可能会遇到精度问题。

本文总结了大模型训练过程中常见精度问题及通用的精度问题定位方法，力求帮助用户快速排查精度问题，缩短模型精度问题定位的时间。

### 常见问题归类总结

大模型训练中经常出现各种精度问题，常见的问题现象包括loss无法收敛、loss收敛效果不佳、训练后期loss不收敛、精度溢出、loss下降过程中与标杆无法拟合等；这些精度问题可能是多种来源造成的，包括模型结构、数据集、超参数、前反向计算精度、优化器计算、浮点计算精度、随机性等方面。

当出现精度问题时，可以从这些精度误差的来源进行问题分析。先根据CheckList进行快速的排查，再进行参数、权重对齐，固定随机性和开启确定性计算后，再执行进出问题排查和长稳训练排除。当前阶段本文主要针对有精度标杆的场景介绍精度定位的通用方法，后续将陆续添加无精度标杆下的精度问题定位内容。

## 精度问题定位CheckList

在定位算子精度问题之前，首先要排除其他非算子因素的干扰。结合以往精度定位案例，总结了精度定位前的CheckList。为了在定位过程中少走弯路，用户可先根据CheckList进行快速的排查。

### 网络结构CheckList

#### 通用结构

| **关键参数**      | **说明**                                                     | **检查项**                                                                     |
| ----------------- | ------------------------------------------------------------ |-----------------------------------------------------------------------------|
| num_layers        | transformer层数                                              | 对应Megatron num-layers参数，检查是否一致。                                             |
| num_heads         | transformer中attention heads数量                             | 对应Megatron num-attention-heads参数，检查是否一致。                                    |
| hidden_size       | transformer隐藏层大小                                        | 对应Megatron hidden-size参数，检查是否一致。                                            |
| intermediate_size | Feed-Forward Network的隐藏层大小                             | 对应Megatron中ffn-hidden-size参数，检查是否一致。                                        |
| n_kv_heads        | kv分组数                                                     | 对应Megatron中的num-query-groups，检查是否一致。                                        |
| 正则化函数        | 正则化函数，常见结构有LayerNorm、RMSNorm                     | MindFormers中的无正则化函数配置参数，与各模型论文中配置一致。Megatron中可通过normalization自定义配置，检查是否一致。 |
| rms_norm_eps      | 正则化的epsilon参数                                          | 对应Megatron的layernorm_epsilon，检查是否一致。                                        |
| dropout           | 网络中的dropout                                              | 当前MindSpore开启Dropout时，不能开重计算；若进行精度比对建议双边都关闭，减少随机因素。                         |
| 融合计算          | 常见的融合算子包括FA、ROPE、Norm、SwigLU；部分用户会将Wq、Wk、Wv进行融合计算 | 1. 同硬件下进行精度比对时，若有使用融合算子，则需要保持一致。 <br>2. 不同硬件下进行精度比对时，则重点检查融合计算部分是否有计算差异。    |

#### MOE结构

| **关键参数**             | **说明**                                          | **检查项**                                                   |
| ------------------------ | ------------------------------------------------- | ------------------------------------------------------------ |
| expert_num               | 专家数量                                          | 对应Megatron的num-experts，检查是否一致。                    |
| num_experts_chosen       | 每个token选择专家数目                             | 对应Megatron的moe-router-topk，检查是否一致。                |
| capacity_factor          | 专家容量系数                                      | 对应Megatron的moe_expert_capacity_factor参数，检查是否一致。 |
| aux_loss_factor          | 负载均衡loss贡献因子                              | 开启时，建议小于0.05。若进行精度对齐，不建议开启，与Megatron的loss打印方式不一致。 |
| enable_sdrop             | 是否开启sdrop方式                                 | 建议设置成true;对应Megatron需要设置如下参数：<br>  moe-token-drop-policy: position <br>  moe-pad-expert-input-to-capacity: True |
| router_dense_type        | 决定专家的dense层                                 | MindFormers中可配置，建议使用fp32计算，防止溢出；Megatron中不可配置。 |
| use_fused_ops_topkrouter | 是否使用融合算子进行dispatch以及combine的索引计算 | MindFormers中融合算子，当enbable_sdrop=True时参数才生效，精度对齐建议设置成True。 |
| use_shared_expert_gating | 共享专家网络中是否使用gating系数                  | 检查网络的共享专家是否有gating系数，如果有设置成True。       |

### 优化器CheckList

| **关键参数**       | **说明**               | **检查项**                                                   |
| ------------------ | ---------------------- | ------------------------------------------------------------ |
| adam优化器         | 优化器类型             | 若Megatron使用adam优化器，MindFormers的数学等价实现为AdamW。 |
| eps                | adam优化器极小值参数   | 检查参数是否一致，推荐值1e-8。                               |
| beta1              | adam优化器梯度动量参数 | 检查参数是否一致，推荐值0.9。                                |
| beta2              | adam优化器梯度方差参数 | 检查参数是否一致，推荐值0.95。                               |
| weight_decay       | 权重衰减               | 默认情况下bias及一维权重不进行衰减，检查用户是否有特殊操作。 |
| lr                 | 学习率                 | 在设置了warmup、学习率衰减后，画图查看学习率变化是否一致。   |
| lr_warmup_fraction | 学习率预热步数占比     | 在设置了warmup、学习率衰减后，画图查看学习率变化是否一致。   |
| clip_grad          | 修剪梯度               | 检查参数是否一致，推荐值1.0。                                |
| global_batch_size  | 全局批大小             | 检查参数是否一致，可以通过训练过程中的打印日志检查。         |

### 权重CheckList

| **关键参数**    | **说明**             | **检查项**                                                   |
| --------------- | -------------------- | ------------------------------------------------------------ |
| param_init_type | 权重初始化类型       | MindFormers通常会设置param_init_dtype类型为FP32，这是因为梯度通信类型是跟权重类型一致，控制通信类型为FP32。而Megatron的梯度通信类型默认为FP32，不与权重类型绑定。 |
| init-method-std | 权重随机初始化的分布 | 若使用权重随机初始化，需要检查随机分布中的mean/std等参数是否一致。 |

### 混合精度CheckList

| **关键参数**           | **说明**                                             | **检查项**                                                   |
| ---------------------- |----------------------------------------------------| ------------------------------------------------------------ |
| compute_dtype          | 计算精度                                               | Megatron 设置 `--bf16: true` 则为FP16，否则为FP16。          |
| layernorm_compute_type | LayerNorm/RMSNorm的计算精度                             | Megatron不可配置，需要检查实现是否保持一致。                 |
| softmax_compute_type   | MindSpore使用FA时，内部Softmax固定用FA计算，仅在小算子拼接实现时可配置计算类型。 | Megatron不可配置，需要检查实现是否保持一致。                 |
| rotary_dtype           | 旋转位置编码的计算精度                                        | Megatron不可配置，需要检查实现是否保持一致。                 |
| 各权重计算             | embedding、lm_head等各权重精度计算                          | 由于MindFormers权重初始化需要设置为FP32，而通常计算精度为BF16/FP16，需要检查权重计算前，是否将权重数据类型转为BF16/FP16。 |
| bias add               | 线性层的bias                                           | 线性层若有bias，检查add的计算精度是否一致。                  |
| residual add           | 残差相加                                               | 检查残差的计算精度是否与标杆一致                             |
| loss                   | loss计算模块                                           | 检查整个loss模块的计算精度是否与标杆一致                     |
| 算子高精度模式         | 昇腾算子支持高精度模式                                        | 开启方式： `context.set_context(ascend_config= {"ge_options":{ "global":{ "ge.opSelectImplmode":"high_precision" } } })` |

### 并行策略CheckList

| **关键参数**               | **说明**               | **检查项**                                                   |
| -------------------------- | ---------------------- | ------------------------------------------------------------ |
| data_parallel              | 数据并行               | 并行切分会影响通信行为，切分后引入通信的计算跟单卡计算可能会有细微差异。 |
| model_parallel             | 模型并行               | 并行切分会影响通信行为，切分后引入通信的计算跟单卡计算可能会有细微差异。 |
| pipeline_stage             | 流水并行               | 并行切分会影响通信行为，切分后引入通信的计算跟单卡计算可能会有细微差异。 |
| use_seq_parallel           | 对应Megatron短序列并行 | 并行切分会影响通信行为，切分后引入通信的计算跟单卡计算可能会有细微差异。 |
| enable_parallel_optimizer  | 优化器并行             | 优化器并行MindSpore与PyTorch两个框架的实现方案不同，通信行为不一致。进行精度对齐时，建议关闭。 |
| micro_batch_interleave_num | 多副本并行             | 优化器并行MindSpore与PyTorch两个框架的实现方案不同，进行精度对齐时，建议关闭。 |

### 其他CheckList

| 关键点        | 检查项                                                                                          |
| ------------- |----------------------------------------------------------------------------------------------|
| 数据检查      | 查看数据是否异常，可随机抽取部分数据进行decode、encode检查，查看input与label的位置是否正确对应。                                  |
| 特殊词检查    | 检查bos_token_id、eos_token_id、pad_token_id等特殊ids是否与数据制作时的ids保持一致。                              |
| input_ids校验 | 检查embedding中的inputs_id是否符合0<=inputs_id<vocab_size；若有越界行为，会取脏数据，导致精度异常。                       |
| 溢出检测      | 溢出状态对齐PyTorch方式，建议使用INFNAN_MODE，即export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE。           |
| 图算融合      | 关闭图算融合，即enable_graph_kernel: False。                                                          |
| 训推模板一致  | 若进行SFT训练，需要确认训练推理时使用的输入模板一致。                                                                 |
| 版本检查      | 检查MindSpore、MindFormers、CANN版本是否配套，建议使用最新的配套版本。                                              |
| 与开源差异    | MindFormers中的已支持了主流的开源LLM模型，并经过了较为充分的测试。如果用户基于MindFormers中开源模型进行开发，可以重点排查与MindFormers开源模型的差异。 |

## 精度调试工具介绍

精度定位中，主要使用MindSpore的Dump工具。主要支持O0/O1/O2模式，不同模式下支持的Dump功能不完全相同，需要的配置文件以及生成的数据格式也不同。O0/O1支持host和device模式支持Dump数据格式`.npy`文件；O2仅支持host模式，支持Dump数据格式`.npy`和`.bin`文件。详细介绍参考[Dump功能调试](https://www.mindspore.cn/docs/zh-CN/master/model_train/debug/dump.html)，下面仅简单介绍两种Dump方式。

### O0/O1 图模式Dump方式

MindSpore的Dump工具通过配置JSON文件进行使能，该方式Dump出网络中的所有算子数据，保存tensor及统计信息的statistic.csv表格。以下给出(O0，O1)模式下的全量算子Dump的JSON示例：

```json
{
    "common_dump_settings": {
        "op_debug_mode": 0,
        "dump_mode": 0,
        "path": "/absolute_path",
        "net_name": "ResNet50",
        "iteration": "0|5-8|100-120",
        "saved_data": "tensor",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"],
        "support_device": [0,1,2,3,4,5,6,7]
    },
    "e2e_dump_settings": {
        "enable": true,
        "trans_flag": true
    }
}
```

配置参数的字段含义参考[Dump功能调试](https://www.mindspore.cn/docs/zh-CN/master/model_train/debug/dump.html)。

配置好JSON文件后， 设置Dump环境变量指向配置的JSON文件，需要设置绝对路径：

```shell
export MINDSPORE_DUMP_CONFIG=${JSON_PATH}
```

设置环境变量后，启动程序训练，即可获取相应的Dump数据。

### O2 图模式Dump

该方式Dump出网络中的所有算子数据，保存tensor及统计信息的statistic.csv表格。O2模式下的全量算子Dump的JSON示例如下，

```json
{
    "common_dump_settings": {
        "op_debug_mode": 0,
        "dump_mode": 0,
        "path": "/absolute_path",
        "net_name": "ResNet50",
        "iteration": "0|5-8|100-120",
        "saved_data": "tensor",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"],
        "support_device": [0,1,2,3,4,5,6,7],
        "statistic_category": ["max", "min", "l2norm"],
        "file_format": "npy"
    }
}
```

配置参数的字段含义参考[Dump功能调试](https://www.mindspore.cn/docs/zh-CN/master/model_train/debug/dump.html)。

配置好JSON文件后，设置Dump环境变量指向配置的JSON文件，需要设置绝对路径：

```shell
export MINDSPORE_DUMP_CONFIG=${JSON_PATH}
export MS_ACL_DUMP_CFG_PATH=${JSON_PATH}
```

设置环境变量后，启动程序训练，即可获取相应的Dump数据。

### msprobe工具介绍

**msprobe** 是 MindStudio Training Tools 工具链下精度调试部分的工具包，主要适用于MindSpore动态图场景的精度问题定位。安装过程请参考[工具安装文档](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/01.installation.md)， 主要包括以下几个功能：
| 功能             | 简要说明                                                     | 详细说明                                                     |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 数据采集         | msprobe 工具通过在训练脚本内添加 dump 接口、启动训练的方式采集网络中API或模块的输入输出数据。 | [数据采集使用说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md) |
| 精度比对         | msprobe 工具可以对采集下来的数据，进行精度比对。               | [精度比对使用说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/11.accuracy_compare_MindSpore.md) |
| 精度预检     | MindSpore 动态图精度预检通过扫描昇腾 NPU 上用户训练 MindSpore 模型中的所有 mint API，输出精度情况的诊断和分析。工具以模型中所有 mint API 前反向的 dump 结果为输入，构造相应的 API 单元测试，将 NPU 输出与标杆（CPU 高精度）比对，计算对应的精度指标，从而找出 NPU 中存在精度问题的 mint API。 | [精度预检使用说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/09.accuracy_checker_MindSpore.md) |
| 溢出检测    | msprobe 工具提供溢出检测功能，针对网络中的每一个API进行输出数据的溢出检测。 | [溢出检测使用说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/13.overflow_check_MindSpore.md) |
| 梯度状态监测 | 采集梯度数据并进行梯度相似度比对，可以精准定位出现问题的 step。 | [梯度状态检测使用说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/17.grad_probe.md) |  

具体定位流程请参考[msprobe工具精度比对定位案例](#msprobe工具精度比对定位案例)。  

### 其他介绍

除了上述介绍的全量算子Dump，工具还支持部分数据Dump、溢出Dump、指定条件Dump等。限于篇幅，感兴趣的用户可以参考[Dump功能调试](https://www.mindspore.cn/docs/zh-CN/master/model_train/debug/dump.html)进行配置使用。此外，还提供了TroubleShooter的网络开发调试，可在权重转换、权重比对等场景使用，详细信息参考[TroubleShooter工具介绍](https://gitee.com/mindspore/toolkits/tree/master/troubleshooter)。

## 精度定位通用流程

通过章节[精度问题定位CheckList](#精度问题定位checklist)进行快速的排查。若完成CheckList的检查后，精度问题依然存在且无明显指向时，可通过本章节的精度定位通用流程缩小问题范围排查。当前通用流程主要针对有标杆的场景，下文将以GPU+PyTorch与Ascend+MindSpore精度对比的场景为例，对精度定位流程进行介绍。

问题定位的主要思路有两点：

* 简化训练的场景，基于单卡/单机、小规模模型复现问题。
* 固定随机因素，对比训练过程中与标杆的loss差异，定位出产生精度差异的原因。

模型的训练过程可以分解为如下过程：数据输入、前向计算、loss、反向计算、梯度、优化器权重更新、下一个step。下面将结合如下图的流程，介绍如何对训练各阶段进行排查。

![general_process](./image/general_process.png)

### 阶段一：训练前准备

进行GPU+PyTorch与Ascend+MindSpore精度对比，需要简化场景及固定随机性，再进行问题的复现。主要有如下三个部分：

* 对齐参数，缩小模型规模，单卡/单机复现问题；

* 加载相同的权重训练；

* 每个step训练相同的数据。

#### 参数对齐

在参数对齐环节，部分参数需要特别说明，参考如下设置。其余参数按照原场景设置，保证PyTorch与MindSpore参数一致即可。参数设置说明：

| 参数                 | 参数建议 | 说明                            |
|--------------------| -------- |-------------------------------|
| num_layers         | 2        | 缩小模型规模，方便快速验证在仅有数据并行情况下单卡可运行。 |
| learning_rate_type | constant | 固定学习率，保证与标杆学习率一致。             |
| warmup_steps       | 0        | warmup的步数                     |
| adam-eps           | 1e-8     | 用户若无特殊要求，按照默认值设置。             |
| dropout            | 0        | 关闭随机性参数，如有其他随机性参数均关闭。         |

模型并行、流水并行、序列并行、优化器并行等特性建议先关闭，精度对齐后再逐步增加并行特性。

#### 权重转换

训练过程中，MindSpore与PyTorch加载同一份权重。若是预训练场景，可以使用PyTorch保存一个初始化权重后，转换为MindSpore权重。因为MindSpore的权重名称与PyTorch有差异，权重转换的本质是将PyTorch权重dict中的名字改为MindSpore权重名字以支持MindSpore加载。权重转换参考[权重转换指导](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html)。

MindSpore与PyTorch均支持`bin`格式数据，加载相同的数据集进行训练，保证每个step一致。

#### 固定随机性，开启确定性计算

训练过程中固定随机性，开启确定性计算，方式如下：

* NPU添加如下环境变量：

  ```shell
  export HCCL_DETERMINISTIC=true  # HCCL确定性
  export ASCEND_LAUNCH_BLOCKING=1  # 硬件确定性
  ```

* PyTorch代码，在[pretrain_gpt.py](https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py)中，新增seed_all方法，并在main方法中调用，添加方法如下：

  ```python
  import numpy as np
  import random

  def seed_all(seed=42):
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.use_deterministic_algorithms(True)
      torch.cuda.manual_seed_all(seed)
      torch.cuda.manual_seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.enable = False
      torch.backends.cudnn.benchmark = False

  if __name__ == "__main__":
      seed_all()

      # 原始代码
  ```

* MindSpore代码，在[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py)中，新增seed_all方法，并在main方法中调用，添加方法如下：

  ```python
  import numpy as np
  import random

  from mindspore import context

  def seed_all(seed=42):
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      np.random.seed(seed)
      context.set_context(deterministic="ON")

  def main(config):
      seed_all()

      # 原始代码
  ```

完成上面的准备工作后，启动单卡训练。若问题未复现，则将场景逐步复杂化，如添加相关特性、扩大模型规模等，直至问题复现，从而定位到问题原因。若问题复现，或者需要复现的时间比较久，则可以开启阶段2的问题定位。

### 阶段二：基础问题排查

通过对比第一个step（step1）和第二个step（step2）的loss及local norm，依次排查前向计算、反向计算、优化器计算。

#### step1的loss对比

在固定权重、数据集、随机性后，对比训练第一个step的loss值差异。第一个step的loss值由网络的前向计算获得，若与标杆loss的差异较大，则可判定前向计算存在精度差异，这可能是由于模型结构未对齐、算子精度异常导致。可通过打印或者Dump工具获取MindSpore及PyTorch每层的tensor值。当前工具暂不具备自动比对功能，需要用户人工识别对应关系进行比对。MindSpore Dump工具介绍参考[精度调试工具介绍](#精度调试工具介绍)，PyTorch Dump工具使用可参考[精度工具功能说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/05.data_dump_PyTorch.md)

通过PyTorch的api_stack_dump.pkl文件，及MindSpore的statistc.csv文件找到层的对应关系，初步通过max，min，L2Norm判断输入输出的差异程度。若需要进一步的对比，可以加载相应的npy数据进行详细比对。

#### step1的local norm值对比

local norm反映的某个权重切片在该设备上的梯度平方和，与标杆对比local norm值，可以初步评估反向计算的差异。计算公式如下：

$$
localnorm = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}
$$

其中 $x_1 ， x_2， \cdots， x_n$ 为某一个权重的梯度。MindFormers中支持通过yaml配置打印local norm，配置方式如下所示：

```yaml
# wrapper cell config
runner_wrapper:
  type: MFTrainOneStepCell
  local_norm: True
  scale_sense: 1
  loss_scale_value: 65536
  use_clip_grad: True
```

Megatron中无配置打印local的入参，需要嵌入式修改文件[megatron/core/optimizer/optimizer.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer.py)：

```python
from megatron.training import get_args, print_rank_0

def get_parameters(self):
    params = []
    grad_norm_list = []
    for param_group in self.optimizer.param_groups:
        for param in param_group['params']:
            grad_norm = torch.norm(param.grad, 2)
            grad_norm_list.append(grad_norm ** 2)
            params.append(param)
    # 嵌入式修改
    print_rank_0(f"print torch local norm:")
    print_rank_0(grad_norm_list)
    return params
```

下图是local norm对比的示例，对比权重对应的local norm值。

![local norm](./image/local_norm.png)

可发现在该图示的场景下，model.tok_embeddings.embedding_weight的local norm值差异较大，可重点排查embedding的实现及计算精度等。

Local norm值仅作为反向计算是否正确的初步判断，若要深入对比反向计算，需要通过Dump工具逐层对比MindSpore及PyTorch反向计算值。

#### 优化器计算排查

在step1的loss和local norm对齐的情况下，若step2的loss差异较大，则需要进一步排查优化器计算。

* 首先排查影响梯度更新的参数，如learning rate、优化器参数、weight decay等是否与标杆一致。

* 其次排查优化器计算，步骤如下：
    * 保存PyTorch step1的梯度。

    * 在MindSpore step1加载PyTorch的梯度进行优化器更新。

    * 对比更新后的权重差异或step2的loss值差异。

若有显著差异，则说明优化器更新存在问题，需要进一步针对优化器进行定位。

PyTorch保存权重梯度，以使用apex为例，修改文件[apex.optimizers](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer.py)文件:

```python
import numpy as np

def get_parameters(self):
    params = []
    grad_id = 0
    for param_group in self.optimizer.param_groups:
        for param in param_group['params']:
            params.append(param)
            grad_id += 1
            # 嵌入式修改，将torch的梯度保存为numpy
            np.save(f"xx/grad_{grad_id}.npy", param)
    return params
```

MindFormers加载梯度参考实现，注意，需要用户自行找到MindFormers与PyTorch梯度的对应关系，修改[mindformers/wrapper/wrapper.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/wrapper/wrapper.py)：

```python
class MFTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    ...
    def __init__(self):
        # 嵌入式修改，加载torch的权重
        grad_0 = Tensor(np.load(f"xxx/grad_1.npy"))
        grad_1 = Tensor(np.load(f"xxx/grad_x.npy"))
        ...
        self.grads = [grad_0, grad_1, ..., ]

    def construct(self, *inputs):
        ...
        # 嵌入式修改，将梯度强制替换为torch梯度
         grads = self.grads
        if self.use_clip_grad:
            grads, global_norm = self.clip_grad_norm(grads)
```

以上代码，仅为实现参考，需要根据实际情况进行代码修改。

若排查出优化器计算不存在问题，同时第二个step的loss差异较大，则需要通过Dump方式重新详细对比第一个step的反向计算。

### 阶段三：长稳训练排查

经过上述操作对齐step1和step2的loss及local norm，排查前向计算、反向计算、优化器更新后，启动长稳训练，对比每个step的loss。

#### 权重不更新实验

设置learning rate = 0，即权重不更新，训练1千step；对比loss值及global norm的差异。在当前阶段，由于数据较多，详细对比每个step每个权重的local norm工作量大，因此通过对比global norm来判断反向计算误差。这是一种简单的快速验证前反向计算的方式，若有某个step loss或norm的值差异较大，则单独使用该数据分析前向及反向。注意，global norm在Megatron打印的字段为grad norm。

#### 标杆误差确认

在进行权重更新的训练前，需要先确认标杆误差，即关闭确定性计算，重复跑两次标杆训练，查看标杆自身的误差，作为判断误差是否合理的参考。由于硬件或底层调用算子的差异，训练的计算过程会不可避免的存在一定的误差。MindSpore训练与PyTorch进行loss对比时，若误差在标杆误差范围内，且误差围绕0轴上下波动，则可以认为误差合理。

#### loss发散

设置learning rate > 0，权重更新，进行长稳测试。训练至某个step出现loss差异较大现象，之后训练loss开始发散，如图所示：

![loss1](./image/loss1.png)

在该场景下，可针对突变前后的训练进行排查，可尝试如下排查方式：

* 检查loss突变附近的数据情况，排查是否有异常数据。通过tokenizer将数据decode为文字查看数据是否异常；同时可尝试跳过这批数据进行训练，验证是否由数据导致。

* 检查在突变附近是否有精度溢出情况。

* 可以查看local norm是否有异常，Dump突变step的训练数据，排查计算的突变点，分析是否算子异常输出。

#### loss后期差异较大

长稳测试中，还可能出现训练前期拟合较好，后期收敛loss出现较大差异，如图所示：

![loss2](./image/loss2.png)

在该场景下，可从如下角度进行排查：

* 排查参数是否对齐：重点排查与优化器相关的参数，如优化器类型、learning rate、weight decay等。可通过画图对比训练过程中的learning rate变化是否一致，另外需要确认进行weight decay的权重是否与标杆一致。

* 混合精度排查：通过Dump工具，细致排查计算过程中混合精度是否与标杆一致；

* 若收敛时loss存在差异，但差异很小，如小于1%，可通过评测下游任务进行精度验收。

#### 场景扩展

在完成单卡对齐的情况下，逐步由单卡扩展为多卡测试、集群测试；模型规模、相关特性如模型并行、流水并行、优化器并行等，视情况添加。由简单场景逐步扩展至实际训练的场景，从而排查新增的特性对精度的影响。

### 案例详解

本节将结合实际案例，介绍基于上述的精度定位流程完成精度排查。

#### 问题现象

在128卡集群下训练模型，使用 Ascend+MindSpore 训练与 GPU+PyTorch 训练进行对比，发现训练后期收敛的loss比 GPU+PyTorch 高0.1左右。如图所示，收敛不符合预期：

![loss3](./image/loss3.png)

蓝色线为 Ascend+MindSpore 训练曲线，红色线为 GPU+PyTorch 训练曲线。

#### 问题定位过程

在定位前，先对照CheckList进行检查，确认无误后启动问题的定位。

首先step1的loss对齐确认没问题。对比step1的local norm，计算每个权重的local norm值与标杆的差异，Embedding权重的local norm值与标杆的差异大。

![local norm](./image/local_norm.png)

排查原因为MindFormers使用fp32进行权重初始化，前向计算及反向计算embedding时均使用fp32精度计算；而PyTorch的前向及反向计算均为bf16，由此导致了计算出来的local norm值存在差异。

计算精度对齐后，排查优化器计算也没有问题，开始进行长稳训练对齐。

长稳训练排查将由单卡实验扩展到多卡实验，先设置learning rate=0，即权重不更新。前向计算每个step的loss差异在0.001左右，前向计算误差符合预期。反向计算每个step的global norm差异在0.05左右，反向计算差异不大；初步判断模型迁移代码正确，模型结构一致，前反向计算差异不大。

![loss4](./image/loss4.png)

再权重更新，单卡训练，设置learning rate=1e-5，训练1千step。收敛后期loss有稳定的0.1的差异，复现问题。

![loss5](./image/loss5.png)

进行问题排查。识别如下问题：

* 通过Dump的文件排查，识别训练过程中存在计算精度不一致的地方，并将不一致的地方统一。

* Weight decay实现不一致，用户PyTorch网络所有权重均进行weight decay。MindFormers中bias权重及一维权重默认不进行weight decay。

修复问题后，再次进行实验，训练1万step，loss差异在0轴附近波动，且小于0.03， 精度符合预期，单卡精度对齐。

完成单卡训练后，启动多卡训练测试：设置learning rate=1e-5，训练1千step。训练后期收敛一致，但训练中期存在稳定的0.05误差。

![loss6](./image/loss6.png)

为验证该误差为合理范围内，关闭确定性计算，重复跑两次GPU实验。图中红线为MindSpore训练的曲线，蓝色、绿色线分别是第一次、第二次GPU训练的曲线。在7千step左右训练不稳定处，MindSpore训练的曲线正处于两次GPU训练的曲线之间，说明误差处于合理范围内，问题最终解决。

![loss7](./image/loss7.png)  

### msprobe工具精度比对定位案例

#### 总体定位流程

1. 首先针对MindSpore实现的模型和PyTorch实现的模型进行数据dump，由于两侧代码在API粒度上可能无法完全对齐，因此可以先使用模块级数据dump，进行模块粒度的比对分析。
2. 根据模块级数据的比对结果，找到第一个精度无法对齐的模块，对模块进行更细粒度的API级别数据dump。
3. 对展开后的模块内部数据进行比对分析，确认问题点。

#### 详细步骤

1. **工具配置**  

    配置config.json文件："level"为"L0", 代表为模块级别数据dump， "task"为"tensor"代表采集真实数据。"step" 指定为第1个step（从0开始计数），配置文件字段详见[配置文件介绍](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/02.config_introduction.md)。  

    tensor模式（保存真实数据，需较大磁盘空间）如下：

    ```json
    {
        "task": "tensor",
        "dump_path": "/path/to/dump/data",
        "rank": [],
        "step": [0],
        "level": "L0",
        "tensor":{
            "scope": [],
            "list": [],
            "data_mode": ["all"]
        }
    }
    ```

    统计信息md5模式（需要比较二进制一致场景推荐使用）如下：

    ```json
    {
        "task": "statistics",
        "dump_path": "/path/to/dump/data",
        "rank": [],
        "step": [0],
        "level": "L0",
        "statistics":{
            "scope": [],
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "md5"
        }
    }
    ```  

2. **训练代码配置**  

    找到网络模型的训练代码（需要找到训练迭代循环的train方法），在循环中插入工具的dump接口使能数据dump。

    ```python
    # 首先导入msprobe工具的PrecisionDebugger类
    from msprobe.mindspore import PrecisionDebugger

    # 实例化PrecisionDebugger类，并传入config.json文件路径
    debugger = PrecisionDebugger(config_path="/path/to/config.json")
    ...
    ...
    ...
    net = Net()

    # 在训练循环中，插入工具的dump代码
    def train(net):
        ...

        for data in dataset:
            # 开启数据dump， start中传入实例化的网络模型
            debugger.start(net)
            output = net(data)
            ...
            # 结束数据dump
            debugger.stop()
            # 更新迭代step数
            debugger.step()

    ```

3. **启动训练**  

    训练结束后查看dump数据，dump数据保存在config.json中的dump\_path字段路径中，dump结果中包括以下文件：  
    **dump.json**: 其中包括所有数据的基本信息以及统计信息（shape、 dtype、max、 min、mean、norm等信息，md5模式下会有md5值）。示例如下：

    ```json
    {
    "task": "tensor",
    "level": "L0",
    "dump_data_dir": "/home/dump_file/tensor_ms_L0/step0/rank0/dump_tensor_data",
    "data": {
    "Cell.model.language_model.embedding.word_embeddings.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0": {
    "input_args": [
        {
        "type": "mindspore.Tensor",
        "dtype": "BFloat16",
        "shape": [
        4096,
        1,
        8192
        ],
        "Max": 2.46875,
        "Min": -2.765625,
        "Mean": -0.0001125335693359375,
        "Norm": 2704.0,
        "data_name": "Cell.model.language_model.embedding.word_embeddings.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0.input.0.npy"
        }
    ],
    "input_kwargs": {},
    "output": [
        {
        "type": "mindspore.Tensor",
        "dtype": "BFloat16",
        "shape": [
        1024,
        1,
        8192
        ],
        "Max": 2.46875,
        "Min": -2.515625,
        "Mean": -0.00020885467529296875,
        "Norm": 1448.0,
        "data_name": "Cell.model.language_model.embedding.word_embeddings.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0.output.0.npy"
        }
       ]
      }
     }
    }
    ```  

    **stack.json**: 包括所有前向数据的调用栈信息，可以将数据关联到对应代码行。示例如下：

    ```json
    {
    "Cell.model.language_model.embedding.word_embeddings.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0": [
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 507, in _run_construct, \n output = self._run_forward_hook(inputs, output)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 759, in _complex_call, \n output = self._run_construct(*args, **kwargs)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 747, in __call__, \n return self._complex_call(*args, **kwargs)",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/tensor_parallel/layers.py, line 770, in construct, \n output = self.reduce_scatter_to_sp_region(output_parallel)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 2462, in _backward_hook_construct, \n outputs = self.construct(outputs, **kwargs)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 498, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 745, in __call__, \n return self._run_construct(*args, **kwargs)",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/transformer/language_model.py, line 151, in construct, \n embeddings = self.word_embeddings(input_ids)",
    "File /opt/miniconda3/envs/sxy/lib/python3.9/site-packages/mindspore/nn/cell.py, line 2460, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
    "File /opt/miniconda3/envs/sxy/lib/python3.9/site-packages/mindspore/nn/cell.py, line 498, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
    "File /opt/miniconda3/envs/sxy/lib/python3.9/site-packages/mindspore/nn/cell.py, line 745, in __call__, \n return self._run_construct(*args, **kwargs)",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/transformer/language_model.py, line 391, in construct, \n text_embedding_out = self.embedding(enc_input_ids, enc_position_ids,",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 2460, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 498, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 745, in __call__, \n return self._run_construct(*args, **kwargs)",
    "File /home/38bv3_show/PanGu_ms_show/pangu/gpt_model.py, line 104, in construct, \n lm_output = self.language_model(tokens,",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 2460, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
    "File /opt/miniconda3/envs//lib/python3.9/site-packages/mindspore/nn/cell.py, line 498, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 745, in __call__, \n return self._run_construct(*args, **kwargs)",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/pipeline_parallel/pipeline_cell.py, line 429, in construct, \n return self.model(*inputs)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 757, in _complex_call, \n output = self.construct(*args, **kwargs)",
    "File /opt/miniconda3/envs//lib/python3.9/site-packages/mindspore/nn/cell.py, line 747, in __call__, \n return self._complex_call(*args, **kwargs)",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/pipeline_parallel/schedules.py, line 121, in run_forward, \n output_tensor = model(*input_data, recv_data=None)",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/pipeline_parallel/schedules.py, line 735, in forward_backward_pipelining_without_interleaving, \n micro_input_data = run_forward(*micro_input_data,",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/training.py, line 409, in forward_backward_with_pipelining, \n loss, logits, grads = forward_backward_pipelining_without_interleaving(",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/training.py, line 533, in construct, \n (loss, _), grads = self.forward_backward_func(*inputs_tuple, loss_scale=current_step_loss_scale, **inputs_dict)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 757, in _complex_call, \n output = self.construct(*args, **kwargs)",
    "File /opt/miniconda3/envs/lib/python3.9/site-packages/mindspore/nn/cell.py, line 747, in __call__, \n return self._complex_call(*args, **kwargs)",
    "File /home/38bv3_show/third_party/dynamic-parallel/mindformers/experimental/distri_cores/training.py, line 655, in train, \n loss, is_finite, loss_scale, learning_rate = train_one_step_cell(**data)",
    "File /home/38bv3_show/PanGu_ms_show/pretrain_gpt.py, line 303, in main, \n train(",
    "File /home/38bv3_show/PanGu_ms_show/pretrain_gpt.py, line 316, in <module>, \n main()"
    ]
    }
    ```

    **connstruct.json**: 包括模型的结构信息，记录每个模块或API的父模块 ("level"为"L1"时，该文件内容为空)。 示例如下：

    ```json
    {
    "Cell.model.language_model.embedding.word_embeddings.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0": "Cell.model.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0",
    "Cell.model.GPTModel.forward.0": null,
    "Cell.model.language_model.TransformerLanguageModel.forward.0": "Cell.model.GPTModel.forward.0",
    "Cell.model.language_model.embedding.Embedding.forward.0": "Cell.model.language_model.TransformerLanguageModel.forward.0",
    "Cell.model.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0": "Cell.model.language_model.embedding.Embedding.forward.0",
    "Cell.model.language_model.rotary_pos_emb.RotaryEmbedding.forward.0": "Cell.model.language_model.TransformerLanguageModel.forward.0",
    "Cell.model.language_model.encoder.layers.0.input_norm.FusedRMSNorm.forward.0": "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.ParallelTransformer.forward.0": "Cell.model.language_model.TransformerLanguageModel.forward.0",
    "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0": "Cell.model.language_model.encoder.ParallelTransformer.forward.0",
    "Cell.model.language_model.encoder.layers.0.attention.qkv_proj.forward_impl_.LinearWithGradAccumulationAndAsyncCommunication.forward.0": "Cell.model.language_model.encoder.layers.0.attention.qkv_proj.ColumnParallelLinear.forward.0",
    "Cell.model.language_model.encoder.layers.0.attention.ParallelAttention.forward.0": "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.layers.0.attention.qkv_proj.ColumnParallelLinear.forward.0": "Cell.model.language_model.encoder.layers.0.attention.ParallelAttention.forward.0",
    "Cell.model.language_model.encoder.layers.0.attention.out_proj.forward_impl_.LinearWithGradAccumulationAndAsyncCommunication.forward.0": "Cell.model.language_model.encoder.layers.0.attention.out_proj.RowParallelLinear.forward.0",
    "Cell.model.language_model.encoder.layers.0.attention.out_proj.RowParallelLinear.forward.0": "Cell.model.language_model.encoder.layers.0.attention.ParallelAttention.forward.0",
    "Cell.model.language_model.encoder.layers.0.attention.out_proj.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0": "Cell.model.language_model.encoder.layers.0.attention.out_proj.RowParallelLinear.forward.0",
    "Cell.model.language_model.encoder.layers.0.attn_post_norm.FusedRMSNorm.forward.0": "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.layers.0.hidden_states_dropout.DropoutExt.forward.0": "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.layers.0.post_attention_norm.FusedRMSNorm.forward.0": "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.layers.0.mlp.mapping.forward_impl_.LinearWithGradAccumulationAndAsyncCommunication.forward.0": "Cell.model.language_model.encoder.layers.0.mlp.mapping.ColumnParallelLinear.forward.0",
    "Cell.model.language_model.encoder.layers.0.mlp.ParallelMLP.forward.0": "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.layers.0.mlp.mapping.ColumnParallelLinear.forward.0": "Cell.model.language_model.encoder.layers.0.mlp.ParallelMLP.forward.0",
    "Cell.model.language_model.encoder.layers.0.mlp.projection.forward_impl_.LinearWithGradAccumulationAndAsyncCommunication.forward.0": "Cell.model.language_model.encoder.layers.0.mlp.projection.RowParallelLinear.forward.0",
    "Cell.model.language_model.encoder.layers.0.mlp.projection.RowParallelLinear.forward.0": "Cell.model.language_model.encoder.layers.0.mlp.ParallelMLP.forward.0",
    "Cell.model.language_model.encoder.layers.0.mlp.projection.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0": "Cell.model.language_model.encoder.layers.0.mlp.projection.RowParallelLinear.forward.0",
    "Cell.model.language_model.encoder.layers.0.ffn_post_norm.FusedRMSNorm.forward.0": "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.layers.0.hidden_states_dropout.DropoutExt.forward.1": "Cell.model.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.layers.1.input_norm.FusedRMSNorm.forward.0": "Cell.model.language_model.encoder.layers.1.ParallelTransformerLayer.forward.0",
    "Cell.model.language_model.encoder.layers.1.ParallelTransformerLayer.forward.0": "Cell.model.language_model.encoder.ParallelTransformer.forward.0",
    }
    ```

    **dump\_tensor\_data**: 包括所有数据的npy文件（只在task为“tensor”时存在），完整目录结构如下：

    ```
    ├── dump_path
    │   ├── step0
    │   |   ├── rank0
    │   |   │   ├── dump_tensor_data
    |   |   |   |    ├── Cell.model.language_model.embedding.word_embeddings.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0.output.0.npy
    |   |   |   |    ...
    |   |   |   |    └── Cell.model.language_model.embedding.word_embeddings.reduce_scatter_to_sp_region.ReduceScatterToSequenceParallelRegion.forward.0.input.0.npy  
    │   |   |   ├── dump.json
    │   |   |   ├── stack.json
    │   |   |   └── construct.json
    │   |   ├── rank1
    |   |   |   ├── dump_tensor_data
    |   |   |   |   └── ...
    │   |   |   ├── dump.json
    │   |   |   ├── stack.json
    |   |   |   └── construct.json
    │   |   ├── ...
    │   |   |
    |   |   └── rank7
    │   ├── step1
    │   |   ├── ...
    │   ├── step2
    ```

4. **精度比对**  

    将dump得到的MindSpore侧数据与PyTorch侧数据进行精度比对。首先配置compare.json文件，示例如下：

    ```json
    {
        "npu_path": "/home/dump_file/tensor_l0_ms/step0/rank0/dump.json",
        "bench_path": "/home/dump_file/tensor_l0_pt/step0/rank0/dump.json",
        "stack_path": "/home/dump_file/tensor_l0_ms/step0/rank0/stack.json",
        "is_print_compare_log": true
    }
    ```  

    **npu\_path**为MindSpore侧的dump.json文件路径, **bench\_path**为PyTorch侧的dump.json文件路径, **stack\_path**为MindSpore侧的stack.json文件路径。

    **运行比对命令**  
        详细比对相关命令介绍可参考[精度比对使用说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/11.accuracy_compare_MindSpore.md)。
        当前场景使用跨框架Cell级别比对，需要在比对时使用层映射功能，传入层映射文件, 执行以下命令

    ```shell
    msprobe -f mindspore compare -i ./compare.json -o ./output -lm ./layer_mapping.yaml
    ```

    **-i**传入参数为compare.json文件路径， **-o**传入参数为输出目录，输出目录下会生成精度对比的结果文件。  
    **-lm**传入参数为层映射文件的路径（当前由于MindFormers中的代码结构无法与Megatron中的代码结构完全对应，因此需要将两侧的层通过映射表来进行映射，layer\_mapping.yaml中对两侧不同的模块进行了层名的映射，如下图所示，对于同一个模块的子模块，两侧存在命名不同的情况。以ParallelTransformerLayer为例，在MindFormers中其中一个子模块的层名为attention， 而在megatron中同一层则命名为self\_attention， 因此在比对时需要传入该映射文件）。

    ![layer_mapping_1](./image/layer_mapping_1.png)
    ![layer_mapping_2](./image/layer_mapping_2.png)

    **分析比对结果**  
        在输出目录查看比对结果的csv文件：
        ![compare_result](./image/compare_result.png)
        查看表格后发现第一个产生误差的位置为**Cell.network\_with\_loss.model.language\_model.encoder.layers.9.mlp.mapping.ColumnParallelLinear.backward.0.input.0**，从命名中可以看出这个模块的反向的输入存在精度问题，而上一个模块的反向输出没有精度问题，因此怀疑在这两个模块之间的某些计算存在精度问题。

5. **细粒度数据采集与分析**
  通过上一步的表格分析，我们将精度问题出现的区间锁定在了**Cell.network\_with\_loss.model.language\_model.encoder.layers.9.mlp.projection.RowParallelLinear.backward.0.output.0** 和 **Cell.network\_with\_loss.model.language\_model.encoder.layers.9.mlp.mapping.ColumnParallelLinear.backward.0.input.0** 之间。因此我们继续dump这两个模块之间的所有API级别数据，来继续定位。此时将config.json文件中的“level”字段配置为“mix”（代表同时dump模块级以及API级数据），并在“scope”中配置上述两个模块的名字用于锁定区间。

    ```json
    {
        "task":"tensor",
        "dump_path": "./tensor_mix_ms",
        "rank": [],
        "step": [0],
        "level": "mix",
        "tensor": {
            "scope": ["Cell.network_with_loss.model.language_model.encoder.layers.9.mlp.projection.RowParallelLinear.backward.0"，
            "Cell.network_with_loss.model.language_model.encoder.layers.9.mlp.mapping.ColumnParallelLinear.backward.0"],
            "list":[],
            "data_mode": ["all"]
        }
    }
    ```  

    由于我们已经锁定了区间，因此dump数据个数并不多，先分别查看MindSpore侧的dump.json文件和PyTorch侧的dump.json文件进行分析。

    **MidnSpore**:

    ![ms_mix_dump](./image/ms_mix_dump.png)

    **PyTorch**:

    ![torch_mix_dump](./image/torch_mix_dump.png)

    通过查看json文件我们发现在这两个模块间的反向计算中，MindSpore侧对于swiglu的实现是通过手动实现的3个API拼接而成，而PyTorch侧的swiglu则直接使用了融合API，直接调用了npu\_swiglu进行计算，因此怀疑两边可能会存在精度误差。随后将PyTorch侧的融合api改写成相同api拼接计算，发现精度问题消失。

### msprobe工具精度预检使用流程  

#### 数据采集

首先需要使用数据采集功能将网络中所有的API采集出来，若将"task"配置为"tensor"采集，在预检进行单元测试时就会使用真实数据输入。若将"task"配置为"statistics"采集，则单元测试时会随机生成数据输入，"level"需设置为"L1"代表进行API级别dump。

```json
{
    "task":"tensor",
    "dump_path": "./tensor_l1_ms",
    "rank": [],
    "step": [0],
    "level": "L1",
    "tensor": {
        "scope": [],
        "list":[],
        "data_mode": ["all"]
    }
}  
```

#### 精度预检 (只支持mint API)

使用msprobe工具进行精度预检，命令如下：

```shell
msprobe -f mindspore run_ut -api_info ./tensor_l1_ms/step0/rank0/dump.json -o ./output  
```

#### 精度预检结果分析  

预检执行结果包括 `accuracy_checking_result_{timestamp}.csv` 和 `accuracy_checking_details_{timestamp}.csv` 两个文件。`accuracy_checking_result_{timestamp}.csv` 属于 API 级，标明每个 API 是否通过测试。建议用户先查看 `accuracy_checking_result_{timestamp}.csv` 文件，对于其中没有通过测试的或者特定感兴趣的 API，根据其 API Name 字段在 `accuracy_checking_details_{timestamp}.csv` 中查询其各个输出的达标情况以及比较指标。详细介绍请参见[预检结果说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/09.accuracy_checker_MindSpore.md#4-%E9%A2%84%E6%A3%80%E7%BB%93%E6%9E%9C)。

预检结果示例：  
![accuracy_checking_result](./image/accuracy_checking_result.png)  
