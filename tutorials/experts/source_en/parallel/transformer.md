# Distributed Parallel Training of Transformer Models

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/parallel/transformer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## Overview

In recent years, the number of Transformer-based pre-trained model parameters has been increasing, and the growth of memory in Ascend 910, GPU and other devices is significantly smaller than the growth of model size. Therefore, it has been a very urgent need to perform parallel train on the Transformer model. MindSpore provides a distributed Transformer interface `mindspore.nn.transformer.transformer` that configures each operator used inside the Transformer with a parallel strategy, and the user only needs to configure the global `data_parallel` and ` model_parallel` attributes to complete the configuration of the distributed parallel strategy. It can greatly facilitate users to apply Transformer for distributed training. Currently, distributed training supports Ascend 910 and GPU environments, as summarized below:

- `Transformer` provides a simple parallel configuration to achieve both operator-level parallelism and pipeline parallelism.

> Download complete sample code: [distributed_training_transformer](https://gitee.com/mindspore/docs/tree/r2.0.0-alpha/docs/sample_code/distributed_training_transformer)

The directory structure is as follows:

```text
└─sample_code
    ├─distribute_training_transformer
        ├── dataset.py
        ├── model.py
        ├── parallel_recover_train.py
        ├── parallel_save_ckpt_train.py
        ├── preprocess.py
        ├── rank_table_16pcs.json
        ├── rank_table_2pcs.json
        ├── rank_table_8pcs.json
        ├── run_cluster.sh
        ├── run_parallel_recover_ckpt.sh
        ├── run_parallel_save_ckpt.sh
        ├── run.sh
        └── train.py
```

`rank_table_8pcs.json` and `rank_table_2pcs.json` are networking information files to configure the current multi-card environment. The files `model.py`, `dataset.py` and `train.py` are the scripts that define the data import, network structure and training files. `run.sh` is the execution script.

Using the `Transformer` library in `mindspore.parallel`, the user needs to decide on the inputs for both the parallel configuration and the model to complete the distributed configuration. **Distributed configuration only works in semi-automatic and automatic parallel mode**.

## Parallel Configuration Definition

For the definition and implementation of the network in `Transformer`, we set the corresponding shard strategy for each operator. Users can achieve parallel configuration of `Transformer` network by setting the global parallel configuration according to their needs.

`Transformer` currently defines three main categories of parallel configurations `TransformerOpParallelConfig`, `OpParallelConfig` and `EmbeddingOpParallelConfig`.

The import path for `TransformerOpParallelConfig` is `mindspore.nn.transformer`, and the attributes it can configure are shown below:

- `data_parallel (int)`: Set the number of data parallelism, and the default value is 1.
- `model_parallel (int)`: Set the number of model parallelism, and the default value is 1.
- `pipeline_stage (int)`: Set the number of Pipeline Stages, and the default value is 1.
- `micro_batch_num (int)`: Set the number of input Batch slices, i.e. a Batch is sliced into multiple small batches. The default value is 1.
- `optimizer_shard (bool)`: Whether to enable optimizer parallelism. The default value is False.
- `gradient_aggregation_group (int)`: Optimizer parallelism corresponds to the number of gradient aggregations, and the default value is 4.
- `recompute (bool)`: Whether to enable recalculation. The default value is False.
- `vocab_emb_dp (bool)`: Whether to configure Embedding as data parallelism. The default value is True.

We will discuss their differences next. Now, as an example of training a `Transformer` model with a single-machine eight-card, we set the parallel configuration of the `Transformer` model based on the current number of 8-card. We can set `data_parallel`=1 and `model_parallel`=8 as the basic parallelism configuration. Note that in the case of parallel configuration, `data_parallel` \*`model_parallel` \*`pipeline_stages` <= total number of cards. The **parallel configuration** in the corresponding code is as follows.

```python
import mindspore as ms
from mindspore.nn.transformer import TransformerOpParallelConfig
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
parallel_config = TransformerOpParallelConfig(data_parallel=1, model_parallel=8)
```

## Model Definition

After defining the configuration, we can start constructing a network. Since MindSpore provides the `Transformer`, the user only needs to add an additional `Embedding` layer, an output layer and a loss function. The following describes the configuration of each module in turn.

### Embedding Layer

The Embedding layer in Tranformer consists of two main parts: word vector embedding and location vector embedding.

We provide `VocabEmbedding` as a parallel Embedding layer, which needs to be initialized by passing in `EmbeddingOpParallelConfig`. Unlike `OpParallelConfig`, `EmbeddingOpParallelConfig` has the following attributes:

- `data_parallel`: Set the number of data parallelism, and the default value is 1.
- `model_parallel`: Set the number of model parallelism, and the default value is 1.
- `vocab_emb_dp`: Whether to configure Embedding as data parallelism. The default value is True.

`vocab_emb_dp` is used to distinguish between two parallel modes of `embedding_lookup` operations, `data parallelism` and `row slicing parallelism`. When `vocab_emb_dp` is `True`, the process of embedding lookups will be set to data parallelism with a parallelism of `data_parallel`. When `vocab_emb_dp` is `False`, the embedding weights will be evenly divided by `model_parallel` in dimension 0, which can reduce the storage of variables.

Here we define an `EmbeddingLayer` that sums the query word vector and the position vector. Note that we set the `add` and `dropout` operations here. Since the input tensor size is `[batch_size, seq_length, hidden_size]` and the word vector lookup process is data parallel, we call the `shard` method of the operator to set the parallel strategy of these two operators separately according to the data parallel value `data_parallel` in `OpParallelConfig`. If the user does not set the `shard` method, the default operator parallelism strategy is **data parallism with the degree of parallism as card number**. The corresponding code is shown below:

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.transformer import VocabEmbedding
class EmbeddingLayer(nn.Cell):
    def __init__(self, vocab_size, position_size, embedding_size,
                 parallel_config, dropout_rate=0.1):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = VocabEmbedding(vocab_size=vocab_size,
                                             embedding_size=embedding_size,
                                             parallel_config=parallel_config)
        self.position_embedding = VocabEmbedding(vocab_size=position_size,
                                                 embedding_size=embedding_size,
                                                 parallel_config=parallel_config)
        self.add = ops.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.dropout = nn.Dropout(1 - dropout_rate)
        self.dropout.dropout.shard(((parallel_config.data_parallel, 1, 1),))

    def construct(self, input_ids, input_position):
        word_embedding, word_table = self.word_embedding(input_ids)
        position_embedding, _ = self.position_embedding(input_position)
        embed = self.add(word_embedding, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table
```

Note that we also return the embedding_table of the word embedding as a return value.

### Transformer Layer

There are three interfaces that users can call as the main construction API: `Transformer`, `TransformerEncoder` and `TransformerDecoder`. They both need to pass `TransformerOpParallelConfig` as the configuration for parallel settings. We set the corresponding parallel strategy for the operator used inside `Transformer` according to the parallel configuration configured in `TransformerOpParallelConfig`.

> The method `pipeline_func` sets the `stage` to which each `block` in the transformer belongs, whether to turn on recompute and the fusion flag for optimizer slicing. For example, in the following example, we calculate the `stage` corresponding to the current `block` according to the configuration of even division, `layer_id` and `offset` passed-in (in the `Transformer` interface, the `offset` passed in when instantiating `Encoder` is 0, and the value of `offset` passed in `Decoder` is the number of layers of `Encoder`), the total number of layers of `Encoder_layer` and `Decoder_layer`, the number of specified `pipeline_stage`. By default, if the user does not pass in `lambda_func`, it is also set to evenly divide according to the number of layers.

```python
def pipeline_func(network, layer_id, offset, parallel_config, layers):
    layers_per_stage = 2
    pp_id = max(int(layer_id + offset) / layers_per_stage, 1)
    network.pipeline_stage = int(pp_id)
    print(f"pipeline id is:{pp_id}", flush=True)
```

In the following code, we instantiate the `EmbeddingLayer` defined above and call `set_comm_fusion` to mark its corresponding reverse gradient fusion as group 0, and call the `pipeline_stage` method to set the weight of the corresponding embedding as the 0th `stage`. Place the last `Head` class, a simple `Linear` layer, on the last `stage`. In case the user does not set the operator parallelism strategy in the Linear, the default is data parallelism within the current `stage`.

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.transformer import Transformer, AttentionMask, CrossEntropyLoss
from mindspore.nn import Dense as Linear
class Net(nn.Cell):
    """
      Single Transformer Model
    """
    def __init__(self, batch, src_len, tgt_len, hidden_size, vocab_size,
                 en_layer, de_layer, parallel_config, return_loss=False):
        super(Net, self).__init__()
        self.src_embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_size=hidden_size,
                                            position_size=src_len,
                                            parallel_config=parallel_config.embedding_dp_mp_config)
        self.tgt_embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_size=hidden_size,
                                            position_size=tgt_len,
                                            parallel_config=parallel_config.embedding_dp_mp_config)
        total_layers = en_layer + de_layer + 2
        layers_per_stage = total_layers // parallel_config.pipeline_stage
        self.src_embedding.pipeline_stage = 0
        self.tgt_embedding.pipeline_stage = 0
        self.return_loss = return_loss

        def pipeline_func(network, layer_id, offset, parallel_config, layers):
            pp_id = max(int(layer_id + offset) / layers_per_stage, 1)
            network.pipeline_stage = int(pp_id)
            gradient_aggregation_group = 4
            dis = max(int((layer_id + offset) / gradient_aggregation_group), 1)
            network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
            print(f"pipeline id is:{pp_id}", flush=True)

        self.base1 = Transformer(encoder_layers=en_layer,
                                 decoder_layers=de_layer,
                                 batch_size=batch,
                                 src_seq_length=src_len,
                                 tgt_seq_length=tgt_len,
                                 hidden_size=hidden_size,
                                 num_heads=8,
                                 attention_dropout_rate=0.0,
                                 hidden_dropout_rate=0.0,
                                 lambda_func=pipeline_func,
                                 ffn_hidden_size=hidden_size,
                                 parallel_config=parallel_config)

        self.attention_mask = AttentionMask(seq_length=tgt_len)
        self.head = Linear(in_channels=hidden_size, out_channels=vocab_size, has_bias=False)
        self.head.matmul.shard(((1, 1), (1, 1)))
        self.head.pipeline_stage = parallel_config.pipeline_stage - 1
        self.loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
        self.no_equal = ops.NotEqual().shard(((1, 1), ()))

```

### Defining the Loss Function

MindSpore also provides a cross-quotient loss function `mindspore.nn.transformer.CrossEntroyLoss` that supports parallelism. This function takes an `OpParallelConfig` to configure the parallelism attributes.

`OpParallelConfig` actually contains two attributes `data_parallel` and `model_parallel`. These two attributes allow you to configure the parallel configuration of the loss function.

```python
from mindspore.nn.transformer import CrossEntropyLoss, TransformerOpParallelConfig
parallel_config = TransformerOpParallelConfig()
loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
```

## End-to-end Process

After defining the parallel configuration, model and loss function, we further integrate the above code. Before starting the training, we call `auto_parallel_context` to set the parallelism option and set the parallelism mode to `SEMI_AUTO_PARALLEL`. In the case of pipeline parallelism, MindSpore provides additional configurations to further slice the gradient accumulation variables to the cards of the data parallel dimension to save memory footprint. The process is as follows: first turn on optimizer slicing (`enable_parallel_optimizer=True`).
Then set `parallel_optimizer_config= {"gradient_accumulation_shard":True}` to further slice the accumulated variables during pipeline parallel training to save memory, and will introduce communication operators between each `micro_step` for gradient synchronization. Note that the default corresponding value of `gradient_accumulation_shard` is True. If the user wants to improve the performance, he can set this parameter to False.

```python
import mindspore as ms
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False, full_batch=True, loss_repeated_mean=True, device_num=device_num, enable_parallel_optimizer=True, parallel_optimizer_config = {"gradient_accumulation_shard": gradient_accumulation_shard})
```

The description of `stage_num` is as follows. MindSpore uses `stage_num` to determine whether to enter pipeline parallel training.

- Perform operator-level parallelism by setting `stage_num=1`. Users can configure the parallel policy by setting the `model_parallel` and `data_parallel` attributes in `TransformerOpParallelConfig`.
- In case `stage_num>1` is set, it will enter pipeline parallel mode. In the pipeline parallel mode, you need to set the `pipeline_stage` attribute of each `cell` to assign the `cell` to the corresponding device for execution. In addition, after instantiating the network, we need to call `PipelineCell` again to encapsulate the defined network. The role of this `Cell` is to slice the input of the network into `mirco_batch_num` numbers of small data in order to maximize the use of computational resources. Note that we need to call `net.infer_param_pipeline_stage()` instead of `net.trainable_params()` to get the training weights corresponding to the current device `stage` and the number of cards within the stage of the pipeline is at least 8. A detailed tutorial of the pipeline can be found [here](https://mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/pipeline_parallel.html).

The code of the integrated master file is as follows. Note that the definitions of some parameters are omitted here, and the complete list of parameters can be found in the [use case source code](https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/sample_code/distributed_training_transformer/train.py). The code address of which is given in the beginning of this article.

```python
import argparse
import mindspore as ms
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore.nn.transformer import TransformerOpParallelConfig
import mindspore.communication as D
from mindspore.nn import PipelineCell
from mindspore.nn import AdamWeightDecay
from dataset import ToyDataset, Tokenzier
from model import Net


def set_weight_decay(params):
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def main():
    parser = argparse.ArgumentParser(description="Transformer training")
    parser.add_argument("--distribute",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Run distribute, default is true.")
    parser.add_argument("--micro_batch_num",
                        type=int,
                        default=1,
                        help="The micro batch num.")
    parser.add_argument('--pipeline_stage',
                        required=False,
                        type=int,
                        default=1,
                        help='The pipeline stage number.')
    parser.add_argument('--mp',
                        required=False,
                        type=int,
                        default=1,
                        help='The model parallel way.')
    args_opt = parser.parse_args()

    if args_opt.distribute == 'true':
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        dp = device_num // args_opt.mp // args_opt.pipeline_stage
        print("rank_id is {}, device_num is {}, dp is {}".format(rank_id, device_num, dp))
        gradient_accumulation_shard = dp > 1 and args_opt.pipeline_stage > 1
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
               parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
               full_batch=True, loss_repeated_mean=True,
               device_num=device_num, enable_parallel_optimizer=True,
               parallel_optimizer_config={"gradient_accumulation_shard": gradient_accumulation_shard})
    else:
        dp = 1

    parallel_config = TransformerOpParallelConfig(pipeline_stage=args_opt.pipeline_stage,
                                                  micro_batch_num=args_opt.micro_batch_num,
                                                  model_parallel=args_opt.mp,
                                                  data_parallel=dp)

    net = Net(batch=args_opt.batch_size // args_opt.micro_batch_num if args_opt.pipeline_stage else args_opt.batch_size,
              src_len=args_opt.src_len, tgt_len=args_opt.tgt_len,
              vocab_size=args_opt.vocab_size,
              hidden_size=args_opt.d_model,
              en_layer=args_opt.encoder_layer,
              de_layer=args_opt.decoder_layer,
              parallel_config=parallel_config, return_loss=args_opt.train)

    tokenizer = Tokenzier()
    task = ToyDataset(file_path=args_opt.file_path,
                      tokenizer=tokenizer,
                      seq_length=(args_opt.src_len, args_opt.tgt_len))
    dataset = task.get_dataset(batch_size=args_opt.batch_size)

    if args_opt.pipeline_stage > 1:
        net = PipelineCell(net, args_opt.micro_batch_num)
        param = net.infer_param_pipeline_stage()
        print(f"params is:{param}", flush=True)
        group_params = set_weight_decay(param)
        opt = AdamWeightDecay(group_params, learning_rate=args_opt.lr)
    else:
        group_params = set_weight_decay(net.trainable_params())
        opt = AdamWeightDecay(group_params, learning_rate=args_opt.lr)

    if not args_opt.train:
        model = Model(net)
    else:
        model = Model(net, optimizer=opt)

    callback_size = 1
    ckpt_config = CheckpointConfig(save_checkpoint_steps=callback_size, keep_checkpoint_max=4,
                                      integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="test",
                                    config=ckpt_config)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size), ckpoint_cb]
    model.train(1, dataset, callbacks=callback, dataset_sink_mode=False)

if __name__ == "__main__":
    main()
```

## Preparation

### Downloading the Dataset

- [Download WMT14 En-Fr dataset](http://statmt.org/wmt14/test-full.tgz). If you download unsuccessfully to click the link, please try to download it after copying the link address.

Use `newstest2014-fren-ref.en.sgm` as the training set for this task, combine and clean this dataset. Extract the dataset to the `docs/sample_code/distributed_training_transformer` directory.

### Pre-processing

Executing the following code to pre-process the data will generate the `output` directory in the current directory, which will produce the files `wmt14.en_fr.txt` and `wmt14.fr_en.txt`, each line of which is a sentence pair in French and English. We will use `wmt14.fr_en.txt` as the training data.

```python
python preprocess.py
```

### Configuring the Distributed Environment Variables

When performing distributed training in a bare-metal environment (compared to an on-cloud environment, i.e. with Ascend 910 AI processors locally), you need to configure the networking information file for the current multi-card environment. If you use Huawei cloud environment, you can skip this subsection because the cloud service itself is well configured.

Taking Ascend 910 AI processor as an example, a sample json configuration file for one 8-card environment is as follows. This sample names the configuration file as `rank_table_8pcs.json`. 2-card environment configuration can refer to the `rank_table_2pcs.json` file in the sample code.

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

Among the parameter items that need to be modified according to the actual training environment are:

- `server_count` represents the number of machines involved in the training.
- `server_id` represents IP address of the current machine.
- `device_id` represents the physical serial number of the card, i.e. the actual serial number in the machine where the card is located.
- `device_ip` represents the IP address of the integration NIC. You can execute the command `cat /etc/hccn.conf` on the current machine, and the key value of `address_x` is the NIC IP address.
- `rank_id` represents the logic serial number of card, fixed numbering from 0.

### Calling the Collective Communication Repository

The MindSpore distributed parallel training communication uses the Huawei Collective Communication Library `Huawei Collective Communication Library` (hereinafter referred to as HCCL), which can be found in the package that accompanies the Ascend AI processor. `mindspore.communication.management` encapsulates the collective communication interface provided by HCCL to facilitate user configuration of distributed information.
> HCCL implements multi-machine multi-card communication based on Ascend AI processor. There are some usage restrictions. We list the common ones by using distributed services, and you can check the corresponding usage documentation of HCCL for details.
>
> - Support 1, 2, 4, 8-card device clusters in single machine scenario and 8*n-card device clusters in multi-machine scenario.
> - 0-3 cards and 4-7 cards of each machine are consisted of two clusters respectively. 2-card and 4-card must be connected and do not support cross-group creation of clusters during training.
> - When building a multi-machine cluster, you need to ensure that each machine uses the same switch.
> - The server hardware architecture and operating system needs to be SMP (Symmetrical Multi-Processing) processing mode.

The following is sample code for calling the collection communication repository:

```python
import os
from mindspore.communication import init
import mindspore as ms

if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

where,

- `mode=GRAPH_MODE`: Using distributed training requires specifying the run mode as graph mode (PyNative mode does not support parallelism).
- `device_id`: The physical serial number of the card, i.e. the actual serial number in the machine where the card is located.
- `init`: Enables HCCL communication and completes distributed training initialization operations.

## Running the Script

The scripts required for training is edited above, and the corresponding scripts are called by the command.

The current MindSpore distributed execution uses a single-card, single-process operation, i.e., one process runs on each card, with the number of processes matching the number of used cards. In this case, card 0 is executed in the foreground, while the other cards are executed in the background. Each process creates 1 directory to store log information as well as operator compilation information. The following is an example of a distributed training script by using 8 cards to demonstrate how to run the script.

```bash
#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH RANK_SIZE"
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
RANK_SIZE=$2

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./train.py ./model.py ./dataset.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ./train.py --distribute=true --file_path=${DATA_PATH} --mp=${RANK_SIZE} > train.log$i 2>&1 &
    cd ../
done
rm -rf device0
mkdir device0
cp ./train.py ./model.py ./dataset.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
python ./train.py --distribute=true --file_path=${DATA_PATH} --mp=${RANK_SIZE} > train.log0 2>&1 &
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../
```

The script needs to pass in the variables `DATA_PATH` and `RANK_SIZE`, which indicate the absolute path to the `wmt14.fr_en.txt` dataset and the number of cards, respectively.

Distributed-related environment variables are:

- `RANK_TABLE_FILE`: The path to the network information file.
- `DEVICE_ID`: The actual serial number of the current card on the machine.
- `RANK_ID`: The logical serial number of the current card.

For the rest of the environment variables, please refer to the configuration items in the [Installation Tutorial](https://www.mindspore.cn/install).

The running time is about 5 minutes, main part of which is spent on the compilation of the operators, and the actual training time is within 20 seconds. The user can monitor the task process by `ps -ef | grep python`.

The log files are saved to the `device0`, `device1` ...... directories corresponding to `rank` directory. `env.log` records information about the environment variables, and the results about the Loss part are saved in `train.log`. The example is as follows:

```text
epoch: 1 step: 1, loss is 9.9034
epoch: 1 step: 2, loss is 9.9033
epoch: 1 step: 3, loss is 9.9031
epoch: 1 step: 4, loss is 9.9025
epoch: 1 step: 5, loss is 9.9022
```

## Summary

Distributed parallel training can significantly improve the performance of network training. From actual experiments, the performance of distributed training on Transformer 8-card exceeds 5 times that of a single card. The distributed parallelization process of the network introduces some code and configuration complexity, but the benefits are worth it compared to the performance gains.
