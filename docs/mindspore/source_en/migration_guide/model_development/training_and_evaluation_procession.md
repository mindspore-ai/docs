# Inference and Training Process

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/model_development/training_and_evaluation_procession.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## General Operating Environment Settings

We generally need to set up the operating environment before network training and inference, and a general operating environment configuration is given here.

```python
import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size

def init_env(cfg):
    """Initialize the operating environment."""
    ms.set_seed(cfg.seed)
    # If device_target is set to None, use the framework to get device_target automatically, otherwise use the set one.
    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)

    # Configure operation mode, and support graph mode and PYNATIVE mode
    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg.device_target = ms.get_context("device_target")
    # If running on CPU, not configure multiple-cards environment
    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
        cfg.rank_id = 0

    # Set the card to be used at runtime
    if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
        ms.set_context(device_id=cfg.device_id)

    if cfg.device_num > 1:
        # The init method is used to initialize multiple cards, and does not distinguish between Ascend and GPU. get_group_size and get_rank can only be used after init
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg.device_num != group_size:
            raise ValueError(f"the setting device_num: {cfg.device_num} not equal to the real group_size: {group_size}")
        cfg.rank_id = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config"):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
    else:
        cfg.device_num = 1
        cfg.rank_id = 0
        print("run standalone!", flush=True)
```

cfg is the parameter configuration file. Using this template requires at least the following parameters to be configured.

```yaml
seed: 1
device_target: "None"
context_mode: "graph"  # should be in ['graph', 'pynative']
device_num: 1
device_id: 0
```

The above procedure is just a basic configuration of the operating environment. If you need to add some advanced features, please refer to [set_context](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html#mindspore.set_context).

## Generic Scripting Framework

A generic [script rack](https://gitee.com/mindspore/models/tree/master/utils/model_scaffolding) provided by the models bin is used for:

1. yaml parameter file parsing, parameter obtaining
2. ModelArts unified tool both on the cloud and on-premise

The python files in the src directory are placed in the model_utils directory for use, e.g. [resnet](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/src/model_utils).

## Inference Process

A generic inference process is as follows:

```python
import mindspore as ms
from mindspore.train import Model
from mindspore import nn
from src.model import Net
from src.dataset import create_dataset
from src.utils import init_env
from src.model_utils.config import config

# Initialize the operating environment
init_env(config)
# Constructing dataset objects
dataset = create_dataset(config, is_train=False)
# Network model, task-related
net = Net()
# Loss function, task-related
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# Load the trained parameters
ms.load_checkpoint(config.checkpoint_path, net)
# Encapsulation into Model
model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
# Model inference
res = model.eval(dataset)
print("result:", res, "ckpt=", config.checkpoint_path)
```

Generally, the source code for network construction and data processing will be placed in the `src` directory, and the scripting framework will be placed in the `src.model_utils` directory. For example, you can refer to the implementation in [MindSpore models](https://gitee.com/mindspore/models).

The inference process cannot be encapsulated into a Model for operation sometimes, and then the inference process can be expanded into the form of a for loop. See [ssd inference](https://gitee.com/mindspore/models/blob/master/official/cv/SSD/eval.py).

### Inference Verification

In the model analysis and preparation phase, we get the trained parameters of the reference implementation (in the reference implementation README or for training replication). Since the implementation of the model algorithm is not related to the framework, the trained parameters can be first converted into MindSpore [checkpoint](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) and loaded into the network for inference verification.

Please refer to [resnet network migration](https://www.mindspore.cn/docs/en/master/migration_guide/sample_code.html) for the whole process of inference verification.

## Training Process

A general training process is as follows:

```python
import mindspore as ms
from mindspore.train import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import nn
from src.model import Net
from src.dataset import create_dataset
from src.utils import init_env
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

@moxing_wrapper()
def train_net():
    # Initialize the operating environment
    init_env(config)
    # Constructing dataset objects
    dataset = create_dataset(config, is_train=False)
    # Network model, task-related
    net = Net()
    # Loss function, task-related
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # Optimizer implementation, task-related
    optimizer = nn.Adam(net.trainable_params(), config.lr, weight_decay=config.weight_decay)
    # Encapsulation into Model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    # checkpoint saving
    config_ck = CheckpointConfig(save_checkpoint_steps=dataset.get_dataset_size(),
                                         keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint", config=config_ck)
    # Model training
    model.train(config.epoch, dataset, callbacks=[LossMonitor(), TimeMonitor()])

if __name__ == '__main__':
    train_net()
```

Please refer to [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) for checkpoint saving.

### Distributed Training

The multi-card distributed training process is the same as the single-card training process, except for the distributed-related configuration items and gradient aggregation. It should be noted that multi-card parallelism actually starts multiple python processes on MindSpore, and before MindSpore version 1.8, on Ascend environment, multiple processes need to be started manually.

```shell
if [ $# != 4 ]
then
    echo "Usage: sh run_distribution_ascend.sh [DEVICE_NUM] [START_ID] [RANK_TABLE_FILE] [CONFIG_PATH]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

RANK_TABLE_FILE=$(get_real_path $3)
CONFIG_PATH=$(get_real_path $4)

if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
exit 1
fi

if [ ! -f $CONFIG_PATH ]
then
    echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
exit 1
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

export RANK_SIZE=$1
STRAT_ID=$2
export RANK_TABLE_FILE=$RANK_TABLE_FILE

cd $BASE_PATH
for((i=0; i<${RANK_SIZE}; i++))
do
    export DEVICE_ID=$((STRAT_ID + i))
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cp ../*.py ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./train_parallel$i ||exit
    env > env.log
    python train.py --config_path=$CONFIG_FILE --device_num=$RANK_SIZE > log.txt 2>&1 &
    cd ..
done
```

After MindSpore 1.8, it can be launched with mpirun as well as the GPU.

```shell
if [ $# != 2 ]
then
    echo "Usage: sh run_distribution_ascend.sh [DEVICE_NUM] [CONFIG_PATH]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

CONFIG_PATH=$(get_real_path $2)

if [ ! -f $CONFIG_PATH ]
then
    echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
exit 1
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

export RANK_SIZE=$1

cd $BASE_PATH
mpirun --allow-run-as-root -n $RANK_SIZE python ../train.py --config_path=$CONFIG_FILE --device_num=$RANK_SIZE > log.txt 2>&1 &
```

If on the GPU, you can set which cards to use by `export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. Specifying the card number is not currently supported on Ascend.

Please refer to [Distributed Case](https://www.mindspore.cn/tutorials/experts/en/master/parallel/distributed_case.html) for more details.

## Offline Inference

In addition to the possibility of online reasoning, MindSpore provides many offline inference methods for different environments. Please refer to [Model Inference](https://www.mindspore.cn/tutorials/experts/en/master/infer/inference.html) for details.
