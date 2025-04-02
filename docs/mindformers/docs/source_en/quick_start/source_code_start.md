# Calling Source Code to Start

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_en/quick_start/source_code_start.md)

This section shows how to use MindFormers to quickly pull up a LoRA low-parameter fine-tuning task based on the Llama2-7B model. To use other models and tasks via MindFormers, please read the corresponding [model documentation](https://www.mindspore.cn/mindformers/docs/en/r1.3.2/start/models.html).

## Preparing Weights File

MindFormers provides pre-trained weights and word list files that have been converted for pre-training, fine-tuning and inference. Users can also download the official HuggingFace weights and use them after converting the model weights. For convenience, this file won't go into too much detail about converting the original weights here, but you can refer to the [Llama2 documentation](https://gitee.com/mindspore/mindformers/blob/v1.3.2/docs/model_cards/llama2.md) and [weight conversion](https://www.mindspore.cn/mindformers/docs/en/r1.3.2/function/weight_conversion.html) for more details. Please download the `MindSpore` weights, the converted `.ckpt` file, and the `tokenizer.model` file for subsequent processing.

| Model Name | MindSpore Weights | HuggingFace Weights |
| ------ | ------ | ------ |
| Llama2-7B | [llama2_7b.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt) | [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) |

Word list download link: [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

## Preparing Dataset

1. The dataset file alpaca_data.json used in the fine-tuning process can be obtained at [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

2. Install the fastchat tool. The version must be greater than or equal to 0.2.13.

    ```shell
      pip install fastchat>=0.2.13
    ```

3. Data Preprocessing

    The following command needs to be executed in the MindSpore Transformers code root directory, and replaces {path} below with the local path where the dataset files are stored.

    1. Execute [mindformers/tools/dataset_preprocess/llama/alpaca_converter.py](https://gitee.com/mindspore/mindformers/blob/v1.3.2/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py), and use the fastchat tool to add prompt templates to convert the raw dataset into a multi-round conversation format.

        ```shell
          python mindformers/tools/dataset_preprocess/llama/alpaca_converter.py \
            --data_path /{path}/alpaca_data.json \
            --output_path /{path}/alpaca-data-conversation.json
        ```

        **Parameter descriptions**

        - data_path:   Input the path to the downloaded file.
        - output_path: Save path of the output file.

    2. Execute [mindformers/tools/dataset_preprocess/llama/llama_preprocess.py](https://gitee.com/mindspore/mindformers/blob/v1.3.2/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py), and generate MindRecord data and convert data with prompt templates to MindRecord format.

        ```shell
          python mindformers/tools/dataset_preprocess/llama/llama_preprocess.py \
            --dataset_type qa \
            --input_glob /{path}/alpaca-data-conversation.json \
            --model_file /{path}/tokenizer.model \
            --seq_length 4096 \
            --output_file /{path}/alpaca-fastchat4096.mindrecord
        ```

        **Parameter descriptions**

        - dataset_type: Preprocessed data types. The options include "wiki" and "qa."
            - "wiki" is used to process the Wikitext2 dataset, which is suitable for the pre-training and evaluation stages.
            - "qa" is used to process the Alpaca dataset, converting it into a question-answer format, which is suitable for the fine-tuning stage.
            For other dataset conversion scripts, please refer to the corresponding  [model documentation](https://www.mindspore.cn/mindformers/docs/en/r1.3.2/start/models.html).
        - input_glob:   Path to the converted alpaca file.
        - model_file:   Path to the model tokenizer.model file.
        - seq_length:   Sequence length of the output data.
        - output_file:  Save path of the output file.

    3. The console outputs the following, proving that the format conversion was successful.

        ```shell
          # Console outputs
          Transformed 52002 records.
          Transform finished, output files refer: {path}/alpaca-fastchat4096.mindrecord
        ```

## Initiating Fine-tuning

In the MindFormers root directory, use the `run_mindformer.py` unified script to pull up tasks:

- Specify the `config` path `configs/llama2/lora_llama2_7b.yaml` via `--config`.
- Specify dataset path `/{path}/alpaca-fastchat4096.mindrecord` via `-train_dataset_dir`.
- Specify the weights file path `/{path}/llama2_7b.ckpt` via `--load_checkpoint`.
- Turn on weight auto-sharding via `--auto_trans_ckpt True`.
- Experiment with eight NPUs and turn on multiprocessing with `--use_parallel True`.
- Set the running mode to `finetune` via `--run_mode finetune`, i.e., fine-tune.

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```

When the following log appears on the console:

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

It indicates that the fine-tuning task is started, the progress can be monitored in the `output/msrun_log/` directory.

For more details on Llama2, and more startup approaches, please refer specifically to the `Llama2` [README](https://gitee.com/mindspore/mindformers/blob/v1.3.2/docs/model_cards/llama2.md#llama-2) documentation for more support.