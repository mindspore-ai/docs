<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} Deprecated
:class: warning

This page belongs to the **Static Graph (GRAPH_MODE) Implementation** section and has been marked as deprecated. New features are primarily being developed in the "r2.0.0 Dynamic Graph Implementation" section. Please refer to the dynamic graph documentation first.
```

# Pretraining

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/static_graph/guide/pre_training.md)

## Overview

Pretraining refers to training a model on large-scale unlabeled data, so that the model can comprehensively capture a wide range of features of a language. A pretrained model can learn knowledge at the vocabulary, syntax, and semantic levels. After fine-tuning, the knowledge is applied in downstream tasks to optimize the performance of specific tasks. The objective of the MindSpore Transformers framework pretraining is to help developers quickly and conveniently build and train pretrained models based on the Transformer architecture.

## Pretraining Procedure of MindSpore Transformers

Based on actual operations, the basic pretraining process can be divided into the following steps:

### 1. Preparing a Dataset

   The pretraining phase of MindSpore Transformers currently supports datasets in both [Megatron format](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#megatron-dataset) and [MindRecord format](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#mindrecord-dataset). Users can prepare the data according to the specific requirements of their tasks.

### 2. Configuring File Preparation

   The pretraining task in MindSpore Transformers is managed through a unified [configuration file](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html), allowing users to flexibly adjust various [training hyperparameters](https://www.mindspore.cn/mindformers/docs/en/master/feature/training_hyperparameters.html). In addition, pretraining performance can be further optimized using features such as [distributed parallel training](https://www.mindspore.cn/mindformers/docs/en/master/feature/parallel_training.html), [memory optimization](https://www.mindspore.cn/mindformers/docs/en/master/feature/memory_optimization.html), and [other training features](https://www.mindspore.cn/mindformers/docs/en/master/feature/other_training_features.html).

### 3. Launching the Training Task

   MindSpore Transformers provides a convenient [one-click script](https://www.mindspore.cn/mindformers/docs/en/master/feature/start_tasks.html) to launch the pretraining task. During training, users can monitor the progress using [logging](https://www.mindspore.cn/mindformers/docs/en/master/feature/logging.html) and [visualization tools](https://www.mindspore.cn/mindformers/docs/en/master/feature/monitor.html).

### 4. Saving a Model

   Checkpoint files can be saved during training or after completion. Currently, MindSpore Transformers supports saving models in [Ckpt format](https://www.mindspore.cn/mindformers/docs/en/master/feature/ckpt.html) or [Safetensors format](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html), which can be used for later tasks such as resuming training or fine-tuning.

### 5. Fault Recovery

   To handle unexpected interruptions during training, MindSpore Transformers includes [training high availability](https://www.mindspore.cn/mindformers/docs/en/master/feature/high_availability.html) such as final-state saving and automatic recovery. It also supports [resuming training from checkpoints](https://www.mindspore.cn/mindformers/docs/en/master/feature/resume_training.html), improving training stability.

## MindSpore Transformers-based Pretraining Practice

Currently, MindSpore Transformers supports mainstream foundation models in the industry. In this practice, Qwen3-32B is used to demonstrate single-node training and multi-node training, respectively. For more training examples and recommended configurations by scenario, see the [Model Library](https://www.mindspore.cn/mindformers/docs/en/master/introduction/models.html).

### Preparing a Dataset

Currently, MindSpore Transformers supports Megatron dataset, which is typically preprocessed and serialized into binary formats (such as `.bin` or `.idx` files). It also comes with a specific indexing mechanism to enable efficient parallel loading and data sharding in distributed cluster environments.

- Dataset download: [WikiText-103](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)

- Tokenizer model download: [tokenizer.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer.json)

### Data Preprocessing

The MindSpore Transformers pre-training stage currently supports [Megatron format datasets](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#megatron-dataset). Users can refer to the [Datasets](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html) section and use the tools provided by MindSpore to convert the original dataset into Megatron format.

To create a Megatron-formatted dataset, two steps are required. First, convert the original text dataset into JSONL format data. Then, use the script provided by MindSpore Transformers to convert the JSONL format data into .bin and .idx files in Megatron format.

- Convert `wiki.train.tokens` to `jsonl` format data

  Users need to **process the `wiki.train.tokens` dataset into a jsonl format file themselves**. For reference, a conversion scheme is provided in the [community issue](https://gitee.com/mindspore/mindformers/issues/ICOKGY). Users need to develop and verify the conversion logic according to their actual needs.

  Below is an example of a JSONL format file:

  ```json
  {"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
  {"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
  ...
  ```

- Convert `jsonl` format data to `bin` format data

  MindSpore Transformers provides a data preprocessing script, `toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py`, for converting raw text data in JSONL format into .bin or .idx files.

  > You need to download the tokenizer file for the [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer.json) model in advance.

  For example:

  ```shell
  python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
    --input /path/to/data.jsonl \
    --output-prefix /path/to/wiki103-megatron \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-dir /path/to/Qwen3-32B # Models of other specifications can be adjusted to the corresponding tokenizer path
  ```

  After the operation is completed, the files `/path/to/wiki103-megatron_text_document.bin` and `/path/to/wiki103-megatron_text_document.idx` will be generated.

  When filling in the dataset path, you need to use `/path/to/wiki103-megatron_text_document` without the suffix.

## Executing a Pretrained Task

### Single-Node Training

Specify the configuration file [pretrain_qwen3_32b_4k.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/pretrain_qwen3_32b_4k.yaml) and after modifying the configuration, then start the [run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py) script in msrun mode to perform 8-device distributed training.

The configuration provided on the warehouse is a 32B model with a large number of parameters, which makes it impossible to directly start pre-training in a single-machine environment. In this example, the model size is reduced to 0.6B to demonstrate single-machine training. Modify the following parameters in the configuration file while keeping the remaining parameters unchanged:

```yaml
# model_config
model:
  model_config:
    hidden_size: 1024
    num_attention_heads: 16
    num_hidden_layers: 28
```

The launch command is as follows:

```shell
cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_32b_4k.yaml \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 2 \
--parallel_config.pipeline_stage 4 \
--parallel_config.micro_batch_num 4"
```

Here:

- `config`: The model configuration file, located in the **config** directory of the **MindSpore Transformers** repository.
- `parallel_config.data_parallel`: Set the number of data parallel.
- `parallel_config.model_parallel`: Set the number of model parallel.
- `parallel_config.pipeline_stage`: Set the number of pipeline parallel.
- `parallel_config.micro_batch_num`: Set the pipeline parallel microbatch size, which should satisfy `parallel_config.micro_batch_num` >= `parallel_config.pipeline_stage` when `parallel_config.pipeline_stage` is greater than 1.

For detailed instructions on launching the training task, refer to [Start Pre-training Task](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/README.md#3-启动预训练任务).

After the task is executed, the **checkpoint** folder is generated in the **mindformers/output** directory, and the model file (`.safetensors`) is saved in this folder.

### Multi-Node Training

If server resources are sufficient, you can launch multi-node training on multiple **Atlas 800T A2 (64G)** machines as shown below.

Execute the following command on each server. Set `master_ip` to the **IP address** of the **master node** (i.e., the server with `Rank 0`), and `node_rank` to the **Rank** index of each node, ranging from `0` to `1023`.

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_32b_4k.yaml" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

> The example code below assumes the **master node IP** is `192.168.1.1` and the current node's **Rank** is `0`. In actual execution, please set `master_ip` to the real **IP address** of the master node, and set `node_rank` to the **Rank** index of the current node.

**Note**: During multi-node distributed training, some performance problems may occur. To ensure the efficiency and stability of the training process, you are advised to optimize and adjust the performance by referring to [Large Model Performance Optimization Guide](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/performance_optimization.html).
