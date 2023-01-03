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

In addition, the training process can be constructed through a functional approach, which is more flexible:

```python
import mindspore as ms
from mindspore import ops, nn
from mindspore.amp import StaticLossScaler, all_finite
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean,\
    _get_parallel_mode, _is_pynative_parallel

class Trainer:
    """A training example with two losses"""
    def __init__(self, net, loss1, loss2, optimizer, train_dataset, loss_scale=1.0, eval_dataset=None, metric=None):
        self.net = net
        self.loss1 = loss1
        self.loss2 = loss2
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.train_data_size = self.train_dataset.get_dataset_size()    # Get the number of training set batches
        self.weights = self.opt.parameters
        # Note that the first parameter of value_and_grad needs to be a graph that needs to be gradient-derived, typically containing a network and a loss. Here it can be a function, or a Cell
        self.value_and_grad = ops.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)

        # Use in the distributed scenario
        self.grad_reducer = self.get_grad_reducer()
        self.loss_scale = StaticLossScaler(loss_scale)
        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.metric = metric
            self.best_acc = 0

    def get_grad_reducer(self):
        grad_reducer = ops.identity
        parallel_mode = _get_parallel_mode()
        # Determine whether it is a distributed scenario, and refer to the above generic runtime environment settings for the distributed scenario settings
        reducer_flag = (parallel_mode in (ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL)) or \
                       _is_pynative_parallel()
        if reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            grad_reducer = nn.DistributedGradReducer(self.weights, mean, degree)
        return grad_reducer

    def forward_fn(self, inputs, labels):
        """Positive network construction. Note that the first output must be the one that requires the gradient at the end"""
        logits = self.net(inputs)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        loss = self.loss_scale.scale(loss)
        return loss, loss1, loss2

    @ms.jit    # jit acceleration, need to meet the requirements of graph mode build, otherwise it will report an error
    def train_single(self, inputs, labels):
        (loss, loss1, loss2), grads = self.value_and_grad(inputs, labels)
        loss = self.loss_scale.unscale(loss)
        grads = self.loss_scale.unscale(grads)
        grads = self.grad_reducer(grads)
        state = all_finite(grads)
        if state:
            self.opt(grads)

        return loss, loss1, loss2

    def train(self, epochs):
        train_dataset = self.train_dataset.create_dict_iterator()
        self.net.set_train(True)
        for epoch in range(epochs):
            # Training an epoch
            for batch, data in enumerate(train_dataset):
                loss, loss1, loss2 = self.train_single(data["image"], data["label"])
                if batch % 100 == 0:
                    print(f"step: [{batch} /{self.train_data_size}] "
                          f"loss: {loss}, loss1: {loss1}, loss2: {loss2}", flush=True)
            # Reason and save the best checkpoint
            if self.run_eval:
                eval_dataset = self.eval_dataset.create_dict_iterator(num_epochs=1)
                self.net.set_train(False)
                self.metric.clear()
                for batch, data in enumerate(eval_dataset):
                    output = self.net(data["image"])
                    self.metric.update(output, data["label"])
                accuracy = self.metric.eval()
                print(f"epoch {epoch}, accuracy: {accuracy}", flush=True)
                if accuracy >= self.best_acc:
                    # Save the best checkpoint
                    self.best_acc = accuracy
                    ms.save_checkpoint(self.net, "best.ckpt")
                    print(f"Updata best acc: {accuracy}")
                self.net.set_train(True)
```

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
