# Pretraining

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/guide/pre_training.md)

## Overview

Pretraining refers to training a model on large-scale unlabeled data, so that the model can comprehensively capture a wide range of features of a language. A pretrained model can learn knowledge at the vocabulary, syntax, and semantic levels. After fine-tuning, the knowledge is applied in downstream tasks to optimize the performance of specific tasks. The objective of the MindSpore Transformers framework pretraining is to help developers quickly and conveniently build and train pretrained models based on the Transformer architecture.

## Pretraining Procedure of MindSpore Transformers

Based on actual operations, the basic pretraining process can be divided into the following steps:

### 1. Preparing a dataset

   The pretraining phase of MindSpore Transformers currently supports datasets in both [Megatron format](https://www.mindspore.cn/mindformers/docs/en/dev/feature/dataset.html#megatron-dataset) and [MindRecord format](https://www.mindspore.cn/mindformers/docs/en/dev/feature/dataset.html#mindrecord-dataset). Users can prepare the data according to the specific requirements of their tasks.

### 2. Configuring File Preparation

   The pretraining task in MindSpore Transformers is managed through a unified [configuration file](https://www.mindspore.cn/mindformers/docs/en/dev/feature/configuration.html), allowing users to flexibly adjust various [training hyperparameters](https://www.mindspore.cn/mindformers/docs/en/dev/feature/training_hyperparameters.html). In addition, pretraining performance can be further optimized using features such as [distributed parallel training](https://www.mindspore.cn/mindformers/docs/en/dev/feature/parallel_training.html), [memory optimization](https://www.mindspore.cn/mindformers/docs/en/dev/feature/memory_optimization.html), and [other training features](https://www.mindspore.cn/mindformers/docs/en/dev/feature/other_training_features.html).

### 3. Launching the Training Task

   MindSpore Transformers provides a convenient [one-click script](https://www.mindspore.cn/mindformers/docs/en/dev/feature/start_tasks.html) to launch the pretraining task. During training, users can monitor the progress using [logging](https://www.mindspore.cn/mindformers/docs/en/dev/feature/logging.html) and [visualization tools](https://www.mindspore.cn/mindformers/docs/en/dev/feature/monitor.html).

### 4. Saving a model

   Checkpoint files can be saved during training or after completion. Currently, MindSpore Transformers supports saving models in [Ckpt format](https://www.mindspore.cn/mindformers/docs/en/dev/feature/ckpt.html) or [Safetensors format](https://www.mindspore.cn/mindformers/docs/en/dev/feature/safetensors.html), which can be used for later tasks such as resuming training or fine-tuning.

### 5. Fault Recovery

   To handle unexpected interruptions during training, MindSpore Transformers includes [high availability features](https://www.mindspore.cn/mindformers/docs/en/dev/feature/high_availability.html) such as final-state saving and automatic recovery. It also supports [resuming training from checkpoints](https://www.mindspore.cn/mindformers/docs/en/dev/feature/resume_training.html), improving training stability.

## MindSpore Transformers-based Pretraining Practice

Currently, MindSpore Transformers supports mainstream foundation models in the industry. In this practice, DeepSeek-V3-671B is used to demonstrate single-node training and multi-node training, respectively.

### Preparing a Dataset

Currently, MindSpore Transformers supports Megatron dataset, which is typically preprocessed and serialized into binary formats (such as `.bin` or `.idx` files). It also comes with a specific indexing mechanism to enable efficient parallel loading and data sharding in distributed cluster environments.

- Dataset download: [wikitext-103](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)

- Tokenizer model download: [tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json?download=true)

### Data Preprocessing

For dataset processing, refer to [Megatron Dataset - Data Preprocessing](https://www.mindspore.cn/mindformers/docs/en/dev/feature/dataset.html#data-preprocessing).

- Generate Megatron BIN Format Files

   Place the dataset file `wiki.train.tokens` and the tokenizer model file `tokenizer.json` under the `../dataset` directory.

   Use the following command to convert the dataset file into BIN format.

   ```shell
   cd $MINDFORMERS_HOME
   python research/deepseek3/wikitext_to_bin.py \
    --input ../dataset/wiki.train.tokens \
    --output-prefix ../dataset/wiki_4096 \
    --vocab-file ../dataset/tokenizer.json \
    --seq-length 4096 \
    --workers 1
   ```

- Build the Megatron BIN Dataset Module

   Run the following command to build the Megatron BIN dataset module.

   ```shell
   pip install pybind11
   cd $MINDFORMERS_HOME/mindformers/dataset/blended_datasets
   make
   ```

   Here, `$MINDFORMERS_HOME` refers to the directory where the **MindSpore Transformers** source code is located.

## Executing a Pretrained Task

### Single-Node Training

Specify the configuration file [pretrain_deepseek3_671b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml) and start the [run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py) script in msrun mode to perform 8-device distributed training.

The default configuration includes large values for parameters such as the number of layers and hidden dimensions, which are intended for large-scale multi-node distributed training. It cannot be directly used for pretraining on a single machine. You will need to modify the configuration as described in [DeepSeek-V3 - Configuration Modification](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/README.md#%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE).

For detailed instructions on launching the training task, refer to [Launch Task](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/README.md#%E6%8B%89%E8%B5%B7%E4%BB%BB%E5%8A%A1). The launch command is as follows:

```shell
cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/deepseek3 \
 --config research/deepseek3/deepseek3_671b/pretrain_deepseek3_1b.yaml"
```

   Here:
   - `register_path`:  The path to the model implementation directory.
   - `config`:         The model configuration file, located in the **config** directory of the **MindSpore Transformers** repository.

After the task is executed, the **checkpoint** folder is generated in the **mindformers/output** directory, and the model file (`.safetensors`) is saved in this folder.

### Multi-Node Training

If server resources are sufficient, you can launch multi-node training on multiple **Atlas 800T A2 (64G)** machines as shown below.

Execute the following command on each server. Set `master_ip` to the **IP address** of the **master node** (i.e., the server with `Rank 0`), and `node_rank` to the **Rank** index of each node, ranging from `0` to `1023`.

```shell
master_ip=192.168.1.1
node_rank=0

cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/deepseek3 \
 --config research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml" \
 1024 8 $master_ip 8118 $node_rank output/msrun_log False 7200
```

> The example code below assumes the **master node IP** is `192.168.1.1` and the current node's **Rank** is `0`. In actual execution, please set `master_ip` to the real **IP address** of the master node, and set `node_rank` to the **Rank** index of the current node.

**Note**: During multi-node distributed training, some performance problems may occur. To ensure the efficiency and stability of the training process, you are advised to optimize and adjust the performance by referring to [Large Model Performance Optimization Guide](https://www.mindspore.cn/mindformers/docs/en/dev/advanced_development/performance_optimization.html).

## More Information

For more training examples of different models, see [the models supported by MindFormers](https://www.mindspore.cn/mindformers/docs/en/dev/introduction/models.html).
