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
  3. 配置yaml文件。如果yaml中模型类、模型Config类、模型Tokenzier类使用了外挂代码，即代码文件在[research](https://gitee.com/mindspore/mindformers/tree/master/research)目录或其他外部目录下，需要修改yaml文件：在相应类的`type`字段下，添加`auto_register`字段，格式为“module.class”（其中“module”为类所在脚本的文件名，“class”为类名。如果已存在，则不需要修改）。

      以[predict_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/master/research/llama3_1/llama3_1_8b/predict_llama3_1_8b.yaml)配置为例，对其中的部分配置项进行如下修改：

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

执行脚本[run_harness.sh](https://gitee.com/mindspore/mindformers/blob/master/toolkit/benchmarks/run_harness.sh)进行评测。

run_harness.sh脚本参数配置如下表：

| 参数               | 类型  | 参数介绍                                                                                           | 是否必须 |
|------------------|-----|------------------------------------------------------------------------------------------------|------|
| `--register_path`| str | 外挂代码所在目录的绝对路径。比如[research](https://gitee.com/mindspore/mindformers/tree/master/research)目录下的模型目录 | 否（外挂代码必填）    |
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

### FAQ

1. 使用Harness进行评测，在加载HuggingFace数据集时，报错`SSLError`：

   参考[SSL Error报错解决方案](https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models)。

   注意：关闭SSL校验存在风险，可能暴露在中间人攻击（MITM）下。仅建议在测试环境或你完全信任的连接里使用。

## 训练后模型进行评测

### 概述

模型在训练过程中或训练结束后，一般会将训练得到的模型权重去跑评测任务，来验证模型的训练效果。本章节介绍了从训练后到评测前的必要步骤，包括：

1. 训练后的分布式权重的处理（单卡训练可忽略此步骤）；
2. 基于训练配置编写评测使用的推理配置文件；
3. 运行简单的推理任务验证上述步骤的正确性；
4. 进行评测任务。

用户可以参考本文档来对自己训练的模型进行评测。

### 分布式权重合并

训练后产生的权重如果是分布式的，需要先将已有的分布式权重合并成完整权重后，再通过在线切分的方式进行权重加载完成推理任务。使用MindSpore Transformers提供的[safetensors权重合并脚本](https://gitee.com/mindspore/mindformers/blob/master/toolkit/safetensors/unified_safetensors.py)，合并后的权重格式为完整权重。

可以按照以下方式填写参数：

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --filter_out_param_prefix "adam_"
```

脚本参数说明：

- src_strategy_dirs：源权重对应的分布式策略文件路径，通常在启动训练任务后默认保存在 output/strategy/ 目录下。分布式权重需根据以下情况填写：

   1. 源权重开启了流水线并行：权重转换基于合并的策略文件，填写分布式策略文件夹路径。脚本会自动将文件夹内的所有 ckpt_strategy_rank_x.ckpt 文件合并，并在文件夹下生成 merged_ckpt_strategy.ckpt。如果已经存在 merged_ckpt_strategy.ckpt，可以直接填写该文件的路径。
   2. 源权重未开启流水线并行：权重转换可基于任一策略文件，填写任意一个 ckpt_strategy_rank_x.ckpt 文件的路径即可。

   注意：如果策略文件夹下已存在 merged_ckpt_strategy.ckpt 且仍传入文件夹路径，脚本会首先删除旧的 merged_ckpt_strategy.ckpt，再合并生成新的 merged_ckpt_strategy.ckpt 以用于权重转换。因此，请确保该文件夹具有足够的写入权限，否则操作将报错。

- mindspore_ckpt_dir：分布式权重路径，请填写源权重所在文件夹的路径，源权重应按 model_dir/rank_x/xxx.safetensors 格式存放，并将文件夹路径填写为 model_dir。
- output_dir：目标权重的保存路径，默认值为 `/new_llm_data/******/ckpt/nbg3_31b/tmp`，即目标权重将放置在 `/new_llm_data/******/ckpt/nbg3_31b/tmp` 目录下。
- file_suffix：目标权重文件的命名后缀，默认值为 "1_1"，即目标权重将按照 *1_1.safetensors 格式查找。
- has_redundancy：合并的源权重是否是冗余的权重，默认为 True。
- filter_out_param_prefix：合并权重时可自定义过滤掉部分参数，过滤规则以前缀名匹配。如优化器参数"adam_"。
- max_process_num：合并最大进程数。默认值：64。

### 推理配置开发

在完成权重文件的合并后，需依据训练配置文件开发对应的推理配置文件。

以Qwen3为例，基于[Qwen3推理配置](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/predict_qwen3.yaml)修改[Qwen3训练配置](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml)：

Qwen3训练配置主要修改点包括：

- run_mode的值修改为"predict"。
- 添加pretrained_model_dir：Hugging Face或ModelScope的模型目录路径，放置模型配置、Tokenizer等文件。
- parallel_config只保留data_parallel和model_parallel。
- model_config中只保留compute_dtype、layernorm_compute_dtype、softmax_compute_dtype、rotary_dtype、params_dtype，和推理配置保持精度一致。
- parallel模块中，只保留parallel_mode和enable_alltoall，parallel_mode的值修改为"MANUAL_PARALLEL"。

### 推理功能验证

在权重和配置文件都准备好的情况下，使用单条数据输入进行推理，检查输出内容是否符合预期逻辑，参考[推理文档](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/guide/inference.md)，拉起推理任务。

例如：

```shell
python run_mindformer.py \
--config configs/qwen3/predict_qwen3.yaml \
--run_mode predict \
--use_parallel False \
--predict_data '帮助我制定一份去上海的旅游攻略'
```

如果输出内容出现乱码或者不符合预期，需要定位精度问题。

1. 检查模型配置正确性

    确认模型结构与训练配置一致。参考训练配置模板使用教程，确保配置文件符合规范，避免因参数错误导致推理异常。

2. 验证权重加载完整性

    检查模型权重文件是否完整加载，确保权重名称与模型结构严格匹配。参考新模型权重转换适配教程，查看权重日志即权重切分方式是否正确，避免因权重不匹配导致推理错误。

3. 定位推理精度问题

    若模型配置与权重加载均无误，但推理结果仍不符合预期，需进行精度比对分析，参考推理精度比对文档，逐层比对训练与推理的输出差异，排查潜在的数据预处理、计算精度或算子问题。

### 使用AISBench进行评测

参考AISBench评测章节，使用AISBench工具进行评测，验证模型精度。