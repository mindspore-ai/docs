## 断点续训

模型训练过程中，在遇到停电宕机、设备内存不足等异常情况导致模型未训练完成的情况下，如果需要从头开始训练，耗时费力，因此在模型训练过程中可以使用断点续训。

先定义基本的网络结构及单步训练函数。

```python
import os
from functools import wraps
from mindspore import nn, Tensor
import mindspore
import numpy as np
from mindspore import save_checkpoint


# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss
```

在MindSpore的函数式场景下实现断点续训，可以定义一个装饰器函数对当前训练过程的异常进行判断和处理，在遇到异常情况时对checkpoint、epoch等信息进行存储。

```python
def save_final_ckpt(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except BaseException as e:
            directory = os.getcwd()
            cur_model_file = "net_epoch_" + str(kwargs.get("epoch")) + "_breakpoint.ckpt"
            save_checkpoint(kwargs.get("model"), os.path.join(directory, cur_model_file))

            cur_opt_file = "net_epoch_" + str(kwargs.get("epoch")) + "_opt.ckpt"
            save_checkpoint(kwargs.get("opt"), os.path.join(directory, cur_opt_file))
            print("====>Exception occurred on epoch {}, {} checkpoint and {} optimizer has been saved.".format(kwargs.get("epoch"), cur_model_file, cur_opt_file))
            raise e
    return wrapper
```

使用装饰器装饰 `train` 函数，以捕获训练过程中的异常并保存断点。

```python
@save_final_ckpt
def train(model, **kwargs):
    model.set_train()
    for step in range(5):
        data, label = Tensor(np.random.rand(64, 1, 28, 28), mindspore.dtype.float32), Tensor(np.random.rand(64,), mindspore.dtype.int32)
        loss = train_step(data, label)
        loss = loss.asnumpy()
    print(f"epoch: {epoch}, loss: {loss:>7f}")
```

在训练异常停止后，使用 `load_final_ckpt` 加载断点。

```python
def load_final_ckpt(model, optimizer, directory=None):
    """Check if there is a exception checkpoint file and load the checkpoint."""
    if not directory:
        directory = os.getcwd()
    files = os.listdir(directory)
    sorted_files = sorted(files, key=lambda file: os.path.getctime(os.path.join(directory, file)), reverse=True)
    model_file_name, opt_file_name = "", ""
    param_dict, opt_dict = {}, {}
    for filename in sorted_files:
        if filename.endswith("opt.ckpt"):
            opt_dict = mindspore.load_checkpoint(filename)
        elif filename.endswith("breakpoint.ckpt"):
            param_dict = mindspore.load_checkpoint(filename)
            model_file_name = filename
        if param_dict and opt_dict:
            mindspore.load_param_into_net(model, param_dict)
            mindspore.load_param_into_net(optimizer, opt_dict)
            initepoch = int(model_file_name.split("_")[2])
            print("====>Network params and Optimizer checkpoint on epoch {} has been loaded successfully.".format(initepoch))
            return model, optimizer, initepoch
    print("====>No model and optimizer checkpoint found, train start.")
    return model, optimizer, 1
```

使用 `resume` 标志位设置是否需要从上次的状态继续训练，如果设置为True，则将保存的断点的网络参数及优化器状态加载到网络中。

```python
epochs = 8
resume = True  # whether to continue training

if resume:
    model, optimizer, initepoch = load_final_ckpt(model, optimizer)
else:
    initepoch = 1
    print("====>Train start.")

for epoch in range(initepoch, epochs):
    train(model=model, opt=optimizer, epoch=epoch)
```

断点保存运行样例。

```python
====>Train start.
epoch: 1, loss: 2.263855
epoch: 2, loss: 2.206145
epoch: 3, loss: 2.145337
====>Exception occurred on epoch 4, net_epoch_4_breakpoint.ckpt checkpoint and net_epoch_4_opt.ckpt optimizer has been saved.
```

续训运行样例。

```python
====>Network params and Optimizer checkpoint on epoch 4 has been loaded successfully.
epoch: 4, loss: 2.081439
epoch: 5, loss: 2.007672
epoch: 6, loss: 1.917388
epoch: 7, loss: 1.795004
```
