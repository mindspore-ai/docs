# 评测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/evaluation.md)

## Harness评测

### 基本介绍

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)是一个开源语言模型评测框架，提供60多种标准学术数据集的评测，支持HuggingFace模型评测、PEFT适配器评测、vLLM推理评测等多种评测方式，支持自定义prompt和评测指标，包含loglikelihood、generate_until、loglikelihood_rolling三种类型的评测任务。基于Harness评测框架对MindSpore Transformers进行适配后，支持加载MindSpore Transformers模型进行评测。

目前已验证过的模型和支持的评测任务如下表所示（其余模型和评测任务正在积极验证和适配中，请关注版本更新）：

| 已验证的模型   | 支持的评测任务                |
|----------|------------------------|
| Llama3   | gsm8k、ceval-valid、mmlu、cmmlu、race、lambada |
| Llama3.1 | gsm8k、ceval-valid、mmlu、cmmlu、race、lambada |
| Qwen2    | gsm8k、ceval-valid、mmlu、cmmlu、race、lambada |

### 安装

Harness支持pip安装和源码编译安装两种方式。pip安装更简单快捷，源码编译安装更便于调试分析，用户可以根据需要选择合适的安装方式。

#### pip安装

用户可以执行如下命令安装Harness（推荐使用0.4.4版本）：

```shell
pip install lm_eval==0.4.4
```

#### 源码编译安装

用户可以执行如下命令编译并安装Harness：

```bash
git clone --depth 1 -b v0.4.4 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### 使用方式

#### 评测前准备

  1. 创建一个新目录，例如名称为`model_dir`，用于存储模型yaml文件。
  2. 在上个步骤创建的目录中，放置模型推理yaml配置文件（predict_xxx_.yaml）。不同模型的推理yaml配置文件所在目录位置，请参考[模型库](../introduction/models.md)。
  3. 配置yaml文件。如果yaml中模型类、模型Config类、模型Tokenzier类使用了外挂代码，即代码文件在[research](https://gitee.com/mindspore/mindformers/tree/dev/research)目录或其他外部目录下，需要修改yaml文件：在相应类的`type`字段下，添加`auto_register`字段，格式为“module.class”（其中“module”为类所在脚本的文件名，“class”为类名。如果已存在，则不需要修改）。

      以[predict_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1/llama3_1_8b/predict_llama3_1_8b.yaml)配置为例，对其中的部分配置项进行如下修改：

        ```yaml
        run_mode: 'predict'       # 设置推理模式
        load_checkpoint: 'model.ckpt'   # 权重路径
        processor:
          tokenizer:
            vocab_file: "tokenizer.model"     # tokenizer路径
            type: Llama3Tokenizer
            auto_register: llama3_tokenizer.Llama3Tokenizer
        ```

        关于每个配置项的详细说明请参考[配置文件说明](../feature/configuration.md)。
  4. 如果使用`ceval-valid`、`mmlu`、`cmmlu`、`race`、`lambada`数据集进行评测，需要将`use_flash_attention`设置为`False`，以`predict_llama3_1_8b.yaml`为例，修改yaml如下：

       ```yaml
       model:
         model_config:
           # ...
           use_flash_attention: False  # 设置为False
           # ...
       ```

#### 评测样例

执行脚本[run_harness.sh](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/run_harness.sh)进行评测。

run_harness.sh脚本参数配置如下表：

| 参数               | 类型  | 参数介绍                                                                                           | 是否必须 |
|------------------|-----|------------------------------------------------------------------------------------------------|------|
| `--register_path`| str | 外挂代码所在目录的绝对路径。比如[research](https://gitee.com/mindspore/mindformers/tree/dev/research)目录下的模型目录 | 否（外挂代码必填）    |
| `--model`        | str | 需设置为 `mf` ，对应为MindSpore Transformers评估策略                                                                  | 是    |
| `--model_args`   | str | 模型及评估相关参数，见下方模型参数介绍                                                                            | 是    |
| `--tasks`        | str | 数据集名称。可传入多个数据集，使用逗号（，）分隔                                                                         | 是    |
| `--batch_size`   | int | 批处理样本数                                                                                         | 否    |

其中，model_args参数配置如下表：

| 参数             | 类型      | 参数介绍               | 是否必须 |
|----------------|---------|--------------------|------|
| `pretrained`   | str     | 模型目录路径             | 是    |
| `max_length`   | int     | 模型生成的最大长度          | 否    |
| `use_parallel` | bool | 开启并行策略(执行多卡评测必须开启) | 否    |
| `tp`           | int     | 张量并行数               | 否    |
| `dp`           | int     | 数据并行数               | 否    |

Harness评测支持单机单卡、单机多卡、多机多卡场景，每种场景的评测样例如下：

1. 单卡评测样例

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir \
       --tasks gsm8k
   ```

2. 多卡评测样例

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=4,dp=1 \
       --tasks ceval-valid \
       --batch_size BATCH_SIZE WORKER_NUM
   ```

   - `BATCH_SIZE`为模型批处理样本数；
   - `WORKER_NUM`为使用计算卡的总数。

3. 多机多卡评测样例

   节点0（主节点）命令：

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
       --tasks lambada \
       --batch_size 2 8 4 192.168.0.0 8118 0 output/msrun_log False 300
   ```

   节点1（副节点）命令：

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
       --tasks lambada \
       --batch_size 2 8 4 192.168.0.0 8118 1 output/msrun_log False 300
   ```

   节点n（副节点）命令：

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
       --tasks lambada \
       --batch_size BATCH_SIZE WORKER_NUM LOCAL_WORKER MASTER_ADDR MASTER_PORT NODE_RANK output/msrun_log False CLUSTER_TIME_OUT
   ```

   - `BATCH_SIZE`为模型批处理样本数；
   - `WORKER_NUM`为所有节点中使用计算卡的总数；
   - `LOCAL_WORKER`为当前节点中使用计算卡的数量；
   - `MASTER_ADDR`为分布式启动主节点的ip；
   - `MASTER_PORT`为分布式启动绑定的端口号；
   - `NODE_RANK`为当前节点的rank id；
   - `CLUSTER_TIME_OUT`为分布式启动的等待时间，单位为秒。

   多机多卡评测需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同。

### 查看评测结果

执行评测命令后，评测结果将会在终端打印出来。以gsm8k为例，评测结果如下，其中Filter对应匹配模型输出结果的方式，n-shot对应数据集内容格式，Metric对应评测指标，Value对应评测分数，Stderr对应分数误差。

| Tasks | Version | Filter           | n-shot | Metric      |   | Value  |   | Stderr |
|-------|--------:|------------------|-------:|-------------|---|--------|---|--------|
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑ | 0.5034 | ± | 0.0138 |
|       |         | strict-match     |      5 | exact_match | ↑ | 0.5011 | ± | 0.0138 |

## FAQ

1. 使用Harness进行评测，在加载HuggingFace数据集时，报错`SSLError`：

   参考[SSL Error报错解决方案](https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models)。

   注意：关闭SSL校验存在风险，可能暴露在中间人攻击（MITM）下。仅建议在测试环境或你完全信任的连接里使用。