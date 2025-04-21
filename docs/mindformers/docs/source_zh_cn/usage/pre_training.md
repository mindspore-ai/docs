# 预训练

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/usage/pre_training.md)

## 概述

预训练是指在大规模无标注数据上训练模型，使其能够全面捕捉语言的广泛特性。通过预训练，模型可以学习到词汇、句法和语义等层面的知识，这些知识在下游任务中通过微调得到应用，从而优化特定任务的性能。MindSpore Transformers框架的预训练目标是帮助开发者快速、便捷地构建和训练基于Transformer架构的预训练模型。

## 预训练的基本操作流程

结合实际操作，预训练的基本流程可以分解为以下步骤：

1. **数据集准备：**
   预训练需要在一个大规模、未标注的文本数据集上进行，这些数据集通常包含来自网络、书籍、文章等多种来源的大量文本。数据集的多样性和规模对模型的泛化能力有很大影响。

2. **选择模型架构：**
   根据任务需求和计算资源，选择合适的模型架构来构建预训练模型。

3. **执行预训练：**
   在准备好的大规模数据集上执行预训练，使用配置好的模型架构和训练配置进行长时间的训练，生成最终的预训练模型权重。

4. **保存模型：**
   训练完成后，将模型权重保存到指定位置。

## 基于MindSpore Transformers的预训练实践

MindSpore Transformers目前已经支持业界主流大模型，该实践流程选择以Llama2-7B和Llama3-70B分别展示[单机训练](#单机训练)和[多机训练](#多机训练)。

### 数据集准备

| 数据集名称   |    适用模型    |   适用阶段   |                                      下载链接                                       |
|:--------|:----------:|:--------:|:-------------------------------------------------------------------------------:|
| Wikitext2 | Llama2-7B  | Pretrain | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip) |
| Wiki103 | Llama3-70B | Pretrain |    [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)     |

### 数据预处理

其中Llama2-7B的数据集处理可参考[Wikitext2数据预处理](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md#数据及权重准备)，Llama3-70B的数据集处理可参考[Wiki103数据预处理](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/README.md#数据集及权重准备)。

## 执行预训练任务

### 单机训练

以Llama2-7B为例，通过指定配置文件[pretrain_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/pretrain_llama2_7b.yaml)以msrun的方式启动[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/run_mindformer.py)脚本，进行8卡分布式训练，启动命令如下：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/pretrain_llama2_7b.yaml \
 --train_dataset_dir /{path}/wiki4096.mindrecord \
 --use_parallel True \
 --run_mode train" 8

 # 参数说明：
 config：            模型的配置文件，文件在MindSpore Transformers代码仓中config目录下
 train_dataset_dir： 训练数据集路径
 use_parallel：      是否开启并行
 run_mode：          运行模式，train：训练，finetune：微调，predict：推理
 ```

任务执行完成后，在mindformers/output目录下，会生成checkpoint文件夹，同时模型文件会保存在该文件夹下。

### 多机训练

以Llama3-70B为例，使用[pretrain_llama3_70b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/llama3_70b/pretrain_llama3_70b.yaml)配置文件，以msrun方式运行[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/run_mindformer.py)执行8机64卡预训练。多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数**MASTER_ADDR**设置为主节点的ip地址，所有节点设置的ip地址相同，不同节点之间仅参数**NODE_RANK**不同，各个参数位置含义参见[msrun启动使用指南](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/msrun_launcher.html)。

```shell
# 节点0，设0节点ip为MASTER_ADDR，作为主节点ip，总共64卡且每个节点8卡
# 节点0、节点1、...节点7 依此修改node_num，比如8机，node_num为0~7。
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/llama3 \
 --config research/llama3/llama3_70b/pretrain_llama3_70b.yaml \
 --train_dataset dataset_dir \
 --use_parallel True \
 --run_mode train" \
 64 8 {MASTER_ADDR} 8118 {node_num} output/msrun_log False 300

 # 参数说明：
 register_path：     模型API的注册路径，是一个包含模型Python文件的目录路径（可以是research目录下模型文件夹的路径）
 config：            模型的配置文件，文件在MindSpore Transformers代码仓中config目录下
 train_dataset_dir： 训练数据集路径
 use_parallel：      是否开启并行
 run_mode：          运行模式，train：训练，finetune：微调，predict：推理
```

**注意**： 在多机分布式训练的过程中，可能会遇到一些性能问题。为了确保训练过程的高效性和稳定性，建议参考[大模型性能调优指南](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/perf_optimize/perf_optimize.html)，进行必要的性能优化和调整。

## 更多信息

更多关于不同模型的训练示例，请访问[MindSpore Transformers已支持模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)。