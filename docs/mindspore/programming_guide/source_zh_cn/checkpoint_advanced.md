## 把自定义信息保存成checkpoint文件

可以使用`save_checkpoint`函数把自定义信息保存成 checkpoint文件，函数声明如下：

```python
def save_checkpoint(save_obj, ckpt_file_name, integrated_save=True,
                    async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM")
```

其中必填的参数有：`save_obj`、`ckpt_file_name`。

下面通过具体示例来说明如何使用每个参数。

### `save_obj`和`ckpt_file_name`参数

**`save_obj`**：可以传入一个  Cell类对象或一个list。
**`ckpt_file_name`**：string类型，表示保存checkpoint文件的名称。

```python
from mindspore import save_checkpoint, Tensor
from mindspore import dtype as mstype
```

1. 传入Cell对象

    ```python
    ​net = LeNet()
    ​save_checkpoint(net, "lenet.ckpt")
    ```

    ​执行后就可以把net中的参数保存成`lenet.ckpt`文件。

2. 传入list对象

    list格式如下：[{"name": param_name, "data": param_data}]，它由一组dict对象组成。

    `param_name`为需要保存对象的名称，`param_data`为需要保存对象的数据，它为Tensor类型。

    ```python
    save_list = [{"name": "lr", "data": Tensor(0.01, mstype.float32)}, {"name": "train_epoch", "data": Tensor(20, mstype.int32)}]
    save_checkpoint(save_list, "hyper_param.ckpt")
    ```

    执行后就可以把`save_list`保存成`hyper_param.ckpt`文件。

### `integrated_save`参数

**`integrated_save`**：bool类型，表示参数是否合并保存，默认为True。在模型并行场景下，Tensor会被切分到不同卡所运行的程序中。如果`integrated_save`设置为True，则这些被切分的Tensor会被合并保存到每个checkpoint文件中，这样checkpoint文件保存的就是完整的训练参数。

```python
save_checkpoint(net, "lenet.ckpt", integrated_save=True)
```

### `async_save`参数

**`async_save`**：bool类型，表示是否开启异步保存功能，默认为False。如果设置为True，则会开启多线程执行写checkpoint文件操作，从而可以并行执行训练和保存任务，在训练大规模网络时会节省脚本运行的总时长。

```python
save_checkpoint(net, "lenet.ckpt", async_save=True)
```

### `append_dict`参数

**`append_dict`**：dict类型，表示需要额外保存的信息，例如：

```python
save_dict = {"epoch_num": 2, "lr": 0.01}
save_checkpoint(net, "lenet.ckpt",append_dict=save_dict)
```

执行后，除了net中的参数，`save_dict`的信息也会保存在`lenet.ckpt`中。

## 加载checkpoint文件对其修改，并重新保存成checkpoint文件

如果想要对checkpoint进行修改，可以使用`load_checkpoint`接口，该接口会返回一个dict。

可以对这个dict进行修改，以便进行后续的操作。

```python
from mindspore import Parameter, Tensor, load_checkpoint, save_checkpoint
# 加载checkpoint文件
param_dict = load_checkpoint("lenet.ckpt")
# 可以通过遍历这个dict，查看key和value
for key, value in param_dict.items():
  # key 为string类型
  print(key)
  # value为parameter类型，使用data.asnumpy()方法可以查看其数值
  print(value.data.asnumpy())

# 拿到param_dict后，就可以对其进行基本的增删操作，以便后续使用

# 1.删除名称为"conv1.weight"的元素
del param_dict["conv1.weight"]
# 2.添加名称为"conv2.weight"的元素，设置它的值为0
param_dict["conv2.weight"] = Parameter(Tensor([0]))
# 3.修改名称为"conv1.bias"的值为1
param_dict["conv2.bias"] = Parameter(Tensor([1]))

# 把修改后的param_dict重新存储成checkpoint文件
save_list = []
# 遍历修改后的dict，把它转化成MindSpore支持的存储格式，存储成checkpoint文件
for key, value in param_dict.items():
  save_list.append({"name": key, "value": value.data})
save_checkpoint(save_list, "new.ckpt")
```
