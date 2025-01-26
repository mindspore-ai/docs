# 评测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/usage/evaluation.md)

## Harness评测

### 基本介绍

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)是一个开源语言模型评测框架，提供60多种标准学术数据集的评测，支持HuggingFace模型评测、PEFT适配器评测、vLLM推理评测等多种评测方式，支持自定义prompt和评测指标，包含loglikelihood、generate_until、loglikelihood_rolling三种类型的评测任务。基于Harness评测框架对MindFormers进行适配后，支持加载MindFormers模型进行评测。

目前已适配的模型和支持的评测任务如下表所示（其余模型和评测任务正在积极适配中，请关注版本更新）：

| 适配的模型     | 支持的评测任务                |
|-----------|------------------------|
| Llama3-8B | Gsm8k、Boolq、Mmlu、Ceval |
| Qwen2-7B  | Gsm8k、Boolq、Mmlu、Ceval                  |

### 安装

Harness支持pip安装和源码编译安装两种方式。pip安装更简单快捷，源码编译安装更便于调试分析，用户可以根据需要选择合适的安装方式。

#### pip安装

用户可以执行如下命令安装Harness：

```shell
pip install lm_eval==0.4.4
```

#### 源码编译安装

用户可以执行如下命令编译并安装Harness：

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout v0.4.4
pip install -e .
```

### 使用方式

#### 查看数据集评测任务

用户可以通过如下命令查看Harness支持的所有评测任务：

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --tasks list
```

#### 启动单卡评测脚本

- 评测前准备

  1. 创建模型目录MODEL_DIR；
  2. 模型目录下须放置yaml配置文件（\*.yaml）、分词器文件（\*_tokenizer.py），获取方式参考[模型库](../start/models.md)中各模型说明文档；
  3. 配置yaml配置文件，参考[配置文件说明](../appendix/conf_files.md)。

        yaml配置样例：

        ```yaml
        run_mode: 'predict'       # 设置推理模式
        model:
          model_config:
            use_past: True
            checkpoint_name_or_path: "model.ckpt"      # 权重路径
        processor:
          tokenizer:
            vocab_file: "tokenizer.model"     # tokenizer路径
        ```

- 执行以下评测命令

   ```shell
   #!/bin/bash

   python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=MODEL_DIR,device_id=0" --tasks TASKS
   ```

   > 注: 执行脚本路径：[eval_with_harness.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_harness.py)

#### 评测参数

Harness主要参数

| 参数            | 类型  | 参数介绍                      | 是否必须 |
|---------------|-----|---------------------------|------|
| `--model`      | str | 须设置为mf，对应为MindFormers评估策略 | 是    |
| `--model_args`  | str | 模型及评估相关参数，见下方模型参数介绍       | 是    |
| `--tasks`       | str | 数据集名称，可传入多个数据集，逗号分割       | 是    |
| `--batch_size` | int | 批处理样本数                    | 否    |
| `--limit`       | int | 每个任务的样本数，多用于功能测试          | 否    |

MindFormers模型参数

| 参数           | 类型   | 参数介绍                              | 是否必须 |
|--------------|------|-----------------------------------|------|
| `pretrained`   | str  | 模型目录路径                            | 是    |
| `use_past`     | bool | 是否开启增量推理，generate_until类型的评测任务须开启 | 否    |
| `device_id`    | int  | 设备id                              | 否    |

### 评测样例

```shell
#!/bin/bash

python eval_with_harness.py --model mf --model_args "pretrained=./llama3-8b,use_past=True" --tasks gsm8k
```

评测结果如下，其中Filter对应匹配模型输出结果的方式，Metric对应评测指标，Value对应评测分数，stderr对应分数误差。

| Tasks | Version | Filter           | n-shot | Metric      |   | Value  |   | Stderr |
|-------|--------:|------------------|-------:|-------------|---|--------|---|--------|
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑ | 0.5034 | ± | 0.0138 |
|       |         | strict-match     |      5 | exact_match | ↑ | 0.5011 | ± | 0.0138 |

## VLMEvalKit评测

### 基本介绍

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
是一款专为大型视觉语言模型评测而设计的开源工具包，支持在各种基准测试上对大型视觉语言模型进行一键评估，无需进行繁重的数据准备工作，让评估过程更加简便。 它支持多种图文多模态评测集和视频多模态评测集，支持多种API模型以及基于PyTorch和HF的开源模型，支持自定义prompt和评测指标。基于VLMEvalKit评测框架对MindFormers进行适配后，支持加载MindFormers中多模态大模型进行评测。

目前已适配的模型和支持的评测数据集如下表所示（其余模型和评测数据集正在积极适配中，请关注版本更新）：

| 适配的模型 | 支持的评测任务                                |
|--|----------------------------------------|
| cogvlm2-llama3-chat-19B | MME、MMBench、COCO Caption、MMMU、Text-VQA |
| cogvlm2-video-llama3-chat | MMBench-Video、MVBench                  |

### 支持特性说明

1. 支持自动下载评测数据集；
2. 支持用户自定义输入多种数据集和模型；
3. 一键生成评测结果。

### 安装

用户可以按照以下步骤进行编译安装：

1. 下载并修改代码：由于开源框架在跑MVBench数据集时存在问题，所以需要使用导入[patch](https://github.com/open-compass/VLMEvalKit/issues/633)的方式修改代码。

    执行以下命令：

    ```bash
    git clone https://github.com/open-compass/VLMEvalKit.git
    cd VLMEvalKit
    git checkout 78a8cef3f02f85734d88d534390ef93ecc4b8bed
    git apply eval.patch
    ```

2. 安装

    共有两种安装方式供大家选择：

   （1） 用于安装当前目录（.）下的Python包（耗时长，易于调试，常用于开发环境）：

    ```bash
    pip install -e .
    ```

    （2） 从[requirements.txt](https://github.com/open-compass/VLMEvalKit/blob/main/requirements.txt)文件中读取依赖列表，并安装这些依赖（耗时短）：

    ```bash
    pip install -r requirements.txt
    ```

### 评测

#### 评测前准备

1. 创建模型目录model_path；
2. 模型目录下须放置yaml配置文件（\*.yaml）、分词器文件（\*_tokenizer.model），获取方式参考[模型库](../start/models.md)中各模型说明文档；
3. 配置yaml配置文件，参考[配置文件说明](../appendix/conf_files.md)。

    yaml配置样例：

    ```yaml
    load_checkpoint: "/{path}/model.ckpt"  # 指定权重文件路径
    model:
      model_config:
        use_past: True                         # 开启增量推理
        is_dynamic: False                       # 关闭动态shape

      tokenizer:
        vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
    ```

#### 启动单卡评测脚本

```shell
#!/bin/bash

python eval_with_vlmevalkit.py \
  --data dataset \
  --model model_name \
  --verbose \
  --work-dir /{path}/evaluate_result \
  --model-path /{path}/model_path \
  --config-path /{path}/config_path
```

执行脚本路径：[eval_with_vlmevalkit.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_vlmevalkit.py)

#### 评测参数

| 参数            | 类型  | 参数介绍                             | 是否必须 |
|---------------|-----|----------------------------------|------|
| `--data`      | str | 数据集名称，可传入多个数据集，空格分割。             | 是    |
| `--model`  | str | 模型名称。                            | 是    |
| `--verbose`       | /   | 输出评测运行过程中的日志。                    | 否    |
| `--work-dir`  | str | 存放评测结果的目录，默认存储在当前目录与模型名称相同的文件夹下。 | 否    |
| `--model-path` | str | 包含模型分词器文件、配置文件的文件夹路径。            | 是    |
| `--config-path`       | str | 模型配置文件路径。                        | 是    |

如果因网络限制，服务器不支持在线下载图文数据集时，可以将本地下载好的以.tsv结尾的数据集文件上传至服务器~/LMUData目录下，进行离线评测。（例如：~/LMUData/MME.tsv 或 ~/LMUData/MMBench_DEV_EN.tsv 或 ~/LMUData/COCO_VAL.tsv）

MMbench-Video数据集评测需要使用gpt-4-turbo模型进行评测打分，请提前准备好相应的apikey。

### 评测样例

```shell
#!/bin/bash

python eval_with_vlmevalkit.py \
  --data COCO_VAL \
  --model cogvlm2-llama3-chat-19B \
  --verbose \
  --work-dir /{path}/evaluate_result \
  --model-path /{path}/cogvlm2_model_path \
  --config-path /{path}/cogvlm2_config_path
```

### 查看评测结果

按照上述方式评估后，在存储评测结果的目录中，找到以.json或以.csv结尾的文件查看评估的结果。

评测样例结果如下，其中`Bleu`和`ROUGE_L`表示评估翻译质量的指标，`CIDEr`表示评估图像描述任务的指标。

```json
{
   "Bleu": [
      15.523950970070652,
      8.971141548228058,
      4.702477458554666,
      2.486860744700995
   ],
   "ROUGE_L": 15.575063213115946,
   "CIDEr": 0.01734615519604295
}
```

## 使用VideoBench数据集进行模型评测

### 基本介绍

[Video-Bench](https://github.com/PKU-YuanGroup/Video-Bench/tree/main) 是首个针对 Video-LLM 的综合评估基准，具有三级能力评估，可以系统地评估模型在视频专属理解、先验知识融入和基于视频的决策能力方面的表现。

### 评测前准备

1. 数据集下载

    下载[Video-Bench中的视频数据](https://huggingface.co/datasets/LanguageBind/Video-Bench)，达到的效果如下所示：

    ```text
    egs/VideoBench/
    ├── Eval_video
    │   └── ActivityNet
    │       └── mp4等文件
    │   └── Driving-decision-making
    │       └── mp4等文件
    |    ...
    ```

2. 文本下载

    下载[Video-Bench中的文本数据](https://github.com/PKU-YuanGroup/Video-Bench/tree/main?tab=readme-ov-file)，达到的效果如下所示：

    ```text
    egs/Video-Bench/
    ├── Eval_QA
    │   └── QA等json文件
    |    ...
    ```

3. 所有问题的正确答案下载

    下载[Video-Bench中的答案数据](https://huggingface.co/spaces/LanguageBind/Video-Bench/resolve/main/file/ANSWER.json)。

### 评测

#### 执行推理脚本，获取推理结果

```shell
    python eval_with_videobench.py \
    --model_path model_path \
    --config_path config_path \
    --dataset_name dataset_name \
    --Eval_QA_root Eval_QA_root \
    --Eval_Video_root Eval_Video_root \
    --chat_conversation_output_folder output
```

执行脚本路径：[eval_with_videobench.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_videobench.py)

**参数说明**

| **参数**                 | **是否必选** | **说明**                                     |
|------------------------|---------|--------------------------------------------|
| model_path             | 是       | 存储模型相关文件的文件夹路径，包含模型配置文件及模型词表文件。            |
| config_path            | 是       | 模型配置文件路径。                                  |
| dataset_name           | 否       | 评测数据子集名称，默认为None，评测VideoBench的所有子集。        |
| Eval_QA_root           | 是       | 存放VideoBench数据集的json文件目录。 |
| Eval_Video_root        | 是       | 存放VideoBench数据集的视频文件目录。                    |
| chat_conversation_output_folder | 否       | 生成结果文件的目录。默认存放在当前目录的Chat_results文件夹下。      |

运行结束后，在chat_conversation_output_folder目录下会生成对话结果文件。

#### 根据生成结果进行评测打分

Video-Bench可以根据模型生成的答案利用ChatGPT或T5进行评估，最终得到13个数据子集的最终分数。

例如：使用ChatGPT进行评估打分：

```shell
python Step2_chatgpt_judge.py \
--model_chat_files_folder ./Chat_results \
--apikey sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
--chatgpt_judge_output_folder ./ChatGPT_Judge

python Step3_merge_into_one_json.py \
--chatgpt_judge_files_folder ./ChatGPT_Judge \
--merge_file ./Video_Bench_Input.json
```

上述评测打分命令中的脚本路径为：[Step2_chatgpt_judge.py](https://github.com/PKU-YuanGroup/Video-Bench/blob/main/Step2_chatgpt_judge.py)、[Step3_merge_into_one_json.py](https://github.com/PKU-YuanGroup/Video-Bench/blob/main/Step3_merge_into_one_json.py)

由于ChatGPT可能会将部分问题的回答视为格式错误，因此需要多次运行Step2_chatgpt_judge.py以确保每个问题都由ChatGPT进行验证。