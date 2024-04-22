# 训练及推理流程

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_zh_cn/migration_guide/model_development/training_and_evaluation.md)

## 通用运行环境设置

我们在进行网络训练和推理前，一般需要先进行运行环境设置，这里给出一个通用的运行环境配置：

```python
import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size

def init_env(cfg):
    """初始化运行时环境."""
    ms.set_seed(cfg.seed)
    # 如果device_target设置是None，利用框架自动获取device_target，否则使用设置的。
    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)

    # 配置运行模式，支持图模式和PYNATIVE模式
    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg.device_target = ms.get_context("device_target")
    # 如果是CPU上运行的话，不配置多卡环境
    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
        cfg.rank_id = 0

    # 设置运行时使用的卡
    if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
        ms.set_context(device_id=cfg.device_id)

    if cfg.device_num > 1:
        # init方法用于多卡的初始化，不区分Ascend和GPU，get_group_size和get_rank方法只能在init后使用
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

其中cfg是参数配置文件，使用此通用模板至少需要配置以下参数：

```yaml
seed: 1
device_target: "None"
context_mode: "graph"  # should be in ['graph', 'pynative']
device_num: 1
device_id: 0
```

上面这个过程只是一个最基本的运行环境配置，如需要添加一些高级的功能，请参考[set_context](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/mindspore/mindspore.set_context.html#mindspore.set_context)。

## 通用脚本架

models仓提供的一个通用的[脚本架](https://gitee.com/mindspore/models/tree/r2.3/utils/model_scaffolding)用于：

1. yaml参数文件解析，参数获取
2. ModelArts云上云下统一工具

一般会将src目录下的python文件放到model_utils目录下进行使用，如[resnet](https://gitee.com/mindspore/models/tree/r2.3/official/cv/ResNet/src/model_utils)。

## 推理流程

一个通用的推理流程如下：

```python
import mindspore as ms
from mindspore.train import Model
from mindspore import nn
from src.model import Net
from src.dataset import create_dataset
from src.utils import init_env
from src.model_utils.config import config

# 初始化运行时环境
init_env(config)
# 构造数据集对象
dataset = create_dataset(config, is_train=False)
# 网络模型，和任务有关
net = Net()
# 损失函数，和任务有关
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 加载训练好的参数
ms.load_checkpoint(config.checkpoint_path, net)
# 封装成Model
model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
# 模型推理
res = model.eval(dataset)
print("result:", res, "ckpt=", config.checkpoint_path)
```

一般网络构造，数据处理等源代码会放到`src`目录下，脚本架会放到`src.model_utils`目录下，具体示例可以参考[MindSpore models](https://gitee.com/mindspore/models)里的实现。

有的时候推理流程无法包成Model进行操作，这时可以将推理流程展开成for循环的形式，可以参考[ssd 推理](https://gitee.com/mindspore/models/blob/r2.3/official/cv/SSD/eval.py)。

### 推理验证

在模型分析与准备阶段，我们会拿到参考实现的训练好的参数（参考实现README里或者进行训练复现）。由于模型算法的实现是和框架没有关系的，训练好的参数可以先转换成MindSpore的[checkpoint](https://www.mindspore.cn/tutorials/zh-CN/r2.3.0rc1/beginner/save_load.html)文件加载到网络中进行推理验证。

整个推理验证的流程请参考[resnet网络迁移](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc1/migration_guide/sample_code.html)。

## 训练流程

一个通用的训练流程如下：

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
    # 初始化运行时环境
    init_env(config)
    # 构造数据集对象
    dataset = create_dataset(config, is_train=False)
    # 网络模型，和任务有关
    net = Net()
    # 损失函数，和任务有关
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # 优化器实现，和任务有关
    optimizer = nn.Adam(net.trainable_params(), config.lr, weight_decay=config.weight_decay)
    # 封装成Model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    # checkpoint保存
    config_ck = CheckpointConfig(save_checkpoint_steps=dataset.get_dataset_size(),
                                         keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint", config=config_ck)
    # 模型训练
    model.train(config.epoch, dataset, callbacks=[LossMonitor(), TimeMonitor()])

if __name__ == '__main__':
    train_net()
```

其中checkpoint保存请参考[保存与加载](https://www.mindspore.cn/tutorials/zh-CN/r2.3.0rc1/beginner/save_load.html)。

此外，可以使用函数式的方法构造训练流程，这个过程更加的灵活：

```python
import mindspore as ms
from mindspore import ops, nn
from mindspore.amp import StaticLossScaler, all_finite

class Trainer:
    """一个有两个loss的训练示例"""
    def __init__(self, net, loss1, loss2, optimizer, train_dataset, loss_scale=1.0, eval_dataset=None, metric=None):
        self.net = net
        self.loss1 = loss1
        self.loss2 = loss2
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.train_data_size = self.train_dataset.get_dataset_size()    # 获取训练集batch数
        self.weights = self.opt.parameters
        # 注意value_and_grad的第一个参数需要是需要做梯度求导的图，一般包含网络和loss。这里可以是一个函数，也可以是Cell
        self.value_and_grad = ms.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)

        # 分布式场景使用
        self.grad_reducer = self.get_grad_reducer()
        self.loss_scale = StaticLossScaler(loss_scale)
        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.metric = metric
            self.best_acc = 0

    def get_grad_reducer(self):
        grad_reducer = ops.identity
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        # 判断是否是分布式场景，分布式场景的设置参考上面通用运行环境设置
        reducer_flag = (parallel_mode != ms.ParallelMode.STAND_ALONE)
        if reducer_flag:
            grad_reducer = nn.DistributedGradReducer(self.weights)
        return grad_reducer

    def forward_fn(self, inputs, labels):
        """正向网络构建，注意第一个输出必须是最后需要求梯度的那个输出"""
        logits = self.net(inputs)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        loss = self.loss_scale.scale(loss)
        return loss, loss1, loss2

    @ms.jit    # jit加速，需要满足图模式构建的要求，否则会报错
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
            # 训练一个epoch
            for batch, data in enumerate(train_dataset):
                loss, loss1, loss2 = self.train_single(data["image"], data["label"])
                if batch % 100 == 0:
                    print(f"step: [{batch} /{self.train_data_size}] "
                          f"loss: {loss}, loss1: {loss1}, loss2: {loss2}", flush=True)
            # 推理并保存最好的那个checkpoint
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
                    # 保存最好的那个checkpoint
                    self.best_acc = accuracy
                    ms.save_checkpoint(self.net, "best.ckpt")
                    print(f"Updata best acc: {accuracy}")
                self.net.set_train(True)
```

### 分布式训练

多卡分布式训练除了分布式相关的配置项和梯度聚合外，其他部分和单卡的训练流程是一样的。需要注意的是多卡并行其实在MindSpore上是起多个python的进程执行的，在MindSpore1.8版本以前，在Ascend环境上，需要手动起多个进程：

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

MindSpore1.8之后，可以和GPU一样使用mpirun启动：

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

如果在GPU上，可以通过`export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`来设置使用哪些卡，Ascend上目前不支持指定卡号。

详情请参考[分布式案例](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0rc1/parallel/distributed_case.html)。

## 离线推理

除了可以在线推理外，MindSpore提供了很多离线推理的方法适用于不同的环境，详情请参考[模型推理](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0rc1/infer/inference.html)。
