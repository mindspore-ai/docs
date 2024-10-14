# 快速启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindformers/docs/source_zh_cn/quick_start/source_code_start.md)

本节展示如何使用MindFormers快速拉起一个基于 Llama2-7B 模型的LoRA低参微调任务。如果想要通过MindFormers使用其他模型和任务，请阅读对应的[模型文档](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/start/models.html)。

## 权重文件准备

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过模型权重转换后进行使用。为了方便起见，这里不对转换原始权重过多赘述，有需要请参考`Llama2`文档以及[权重转换](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/function/weight_conversion.html)了解更多细节。这里请直接下载`MindSpore`权重，下载转换后的`.ckpt`文件以及`tokenizer.model`文件进行后续的处理。

| 模型名称 | MindSpore权重 | HuggingFace权重 |
| ------ | ------ | ------ |
| Llama2-7B | [llama2_7b.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt) | [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) |

词表下载链接：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

## 数据集准备

1. 微调过程中使用的数据集在[数据集下载](https://github.com/tatsu-lab/stanford_alpaca)获得。

2. 数据预处理

    1. 执行[mindformers/tools/dataset_preprocess/llama/alpaca_converter.py](https://gitee.com/mindspore/mindformers/blob/r1.3.0/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py)，使用fastchat工具添加prompt模板，将原始数据集转换为多轮对话格式。

        ```shell

          python alpaca_converter.py \
            --data_path /{path}/alpaca_data.json \
            --output_path /{path}/alpaca-data-conversation.json

          # 参数说明
          data_path:   输入下载的文件路径
          output_path: 输出文件的保存路径

        ```

    2. 执行[mindformers/tools/dataset_preprocess/llama/llama_preprocess.py](https://gitee.com/mindspore/mindformers/blob/r1.3.0/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py)，生成MindRecord数据，将带有prompt模板的数据转换为MindRecord格式。

        ```shell
          # 此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9
          python llama_preprocess.py \
            --dataset_type qa \
            --input_glob /{path}/alpaca-data-conversation.json \
            --model_file /{path}/tokenizer.model \
            --seq_length 4096 \
            --output_file /{path}/alpaca-fastchat4096.mindrecord

          # 参数说明
          dataset_type: 预处理数据类型
          input_glob:   转换后的alpaca的文件路径
          model_file:   模型tokenizer.model文件路径
          seq_length:   输出数据的序列长度
          output_file:  输出文件的保存路径
        ```

        控制台输出如下内容，证明格式转换成功。

        ```shell

          # 控制台输出
          Transformed 52002 records.
          Transform finished, output files refer: {path}/alpaca-fastchat4096.mindrecord
        ```

## 启动微调

使用`run_mindformer.py`统一脚本拉起任务：

- 通过 `--config` 指定`config`路径 `configs/llama2/lora_llama2_7b.yaml`。
- 通过 `--train_dataset_dir` 指定数据集路径  `/{path}/alpaca-fastchat4096.mindrecord`。
- 通过 `--load_checkpoint` 指定权重文件路径 `/{path}/llama2_7b.ckpt`。
- 通过 `--auto_trans_ckpt True` 打开权重自动切分功能。
- 使用八块NPU进行实验，通过 `--use_parallel True` 开启多进程工作。
- 通过 `--run_mode finetune` 设定运行模式为 `finetune`，即进行微调。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```

当控制台出现如下日志时：

```shell
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:0, log file:output/msrun_log/worker_0.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:1, log file:output/msrun_log/worker_1.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:2, log file:output/msrun_log/worker_2.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:3, log file:output/msrun_log/worker_3.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:4, log file:output/msrun_log/worker_4.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:5, log file:output/msrun_log/worker_5.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:6, log file:output/msrun_log/worker_6.log. Environment variable [RANK_ID] is exported.
[mindspore/parallel/cluster/process_entity/_api.py:224] Start worker process with rank id:7, log file:output/msrun_log/worker_7.log. Environment variable [RANK_ID] is exported.
```

说明启动微调成功。

## 说明

关于Llama2更多细节，以及更多的启动方式，请具体参考`Llama2` 的 [README](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#llama-2)文档获取更多支持。
