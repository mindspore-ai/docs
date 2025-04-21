# Pretraining

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_en/usage/pre_training.md)

## Overview

Pretraining refers to training a model on large-scale unlabeled data, so that the model can comprehensively capture a wide range of features of a language. A pretrained model can learn knowledge at the vocabulary, syntax, and semantic levels. After fine-tuning, the knowledge is applied in downstream tasks to optimize the performance of specific tasks. The objective of the MindSpore Transformers framework pretraining is to help developers quickly and conveniently build and train pretrained models based on the Transformer architecture.

## Procedure

Based on actual operations, the basic pretraining process can be divided into the following steps:

1. **Preparing a dataset:**
   Prepare a large-scale unlabeled text dataset for pretraining. Such datasets contain a large amount of text from multiple sources, such as networks, books, and articles. The diversity and scale of datasets have a great impact on the generalization capability of models.

2. **Selecting a model architecture:**
   Select a proper model architecture to build a pretrained model based on task requirements and computing resources.

3. **Pretraining:**
   Perform pretraining with the prepared large-scale dataset and use the configured model architecture and training configuration to perform long-time training to generate the final pretrained model weight.

4. **Saving a model:**
   After the training is complete, save the model weight to the specified location.

## MindSpore Transformers-based Pretraining Practice

Currently, MindSpore Transformers supports mainstream foundation models in the industry. In this practice, Llama2-7B and Llama3-70B are used to demonstrate [Single-Node Training](#single-node-training) and [Multi-Node Training](#multi-node-training), respectively.

### Preparing a Dataset

| Dataset  |    Applicable Model   |   Applicable Phase  |                                      Download Link                                      |
|:--------|:----------:|:--------:|:-------------------------------------------------------------------------------:|
| Wikitext2 | Llama2-7B  | Pretrain | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip) |
| Wiki103 | Llama3-70B | Pretrain |    [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)     |

### Data Preprocessing

For details about how to process the Llama2-7B and Llama3-70B datasets, see [the Wikitext2 data preprocessing](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md) and [the Wiki103 data preprocessing](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/README.md), respectively.

## Executing a Pretrained Task

### Single-Node Training

Take Llama2-7B as an example. Specify the configuration file [pretrain_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/pretrain_llama2_7b.yaml) and start the [run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/run_mindformer.py) script in msrun mode to perform 8-device distributed training. The startup command is as follows:

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/pretrain_llama2_7b.yaml \
 --train_dataset_dir /{path}/wiki4096.mindrecord \
 --use_parallel True \
 --run_mode train" 8

 # Parameters:
 config:            model configuration file, which is stored in the config directory of the MindSpore Transformers code repository.
 train_dataset_dir: path of the training dataset.
 use_parallel:      specifies whether to enable parallelism.
 run_mode:          running mode. The value can be train, finetune, or predict (inference).
 ```

After the task is executed, the **checkpoint** folder is generated in the **mindformers/output** directory, and the model file is saved in this folder.

### Multi-Node Training

Take Llama3-70B as an example. Use the [pretrain_llama3_70b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/llama3_70b/pretrain_llama3_70b.yaml) configuration file to run [run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/run_mindformer.py) in msrun mode to perform 8-node 64-device pretraining. To perform distributed training on a multi-node multi-device script, you need to run the script on different nodes and set the **MASTER_ADDR** parameter to the IP address of the primary node. The IP addresses of all nodes are the same, and only the values of **NODE_RANK** are different for different nodes. For details about the parameter positions, see [msrun Launching Guide](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/parallel/msrun_launcher.html).

```shell
# Node 0: Set the IP address of node 0 to the value of MASTER_ADDR, which is used as the IP address of the primary node. There are 64 devices in total with 8 devices for each node.
# Change the value of node_num for nodes 0 to 7 in sequence. For example, if there are eight nodes, the value of node_num ranges from 0 to 7.
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/llama3 \
 --config research/llama3/llama3_70b/pretrain_llama3_70b.yaml \
 --train_dataset dataset_dir \
 --use_parallel True \
 --run_mode train" \
 64 8 {MASTER_ADDR} 8118 {node_num} output/msrun_log False 300

 # Parameters:
 register_path:     The registered path of the model API is a directory path that contains Python scripts of the model (can be the path of the model folder in the 'research' directory).
 config:            model configuration file, which is stored in the config directory of the MindSpore Transformers code repository.
 train_dataset_dir: path of the training dataset.
 use_parallel:      specifies whether to enable parallelism.
 run_mode:          running mode. The value can be train, finetune, or predict (inference).
```

**Note**: During multi-node distributed training, some performance problems may occur. To ensure the efficiency and stability of the training process, you are advised to optimize and adjust the performance by referring to [Large Model Performance Optimization Guide](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/perf_optimize/perf_optimize.html).

## More Information

For more training examples of different models, see [the models supported by MindFormers](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/start/models.html).
