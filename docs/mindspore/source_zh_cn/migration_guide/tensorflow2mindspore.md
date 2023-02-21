# TensorFlow模型转换MindSpore模型文件方法

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/tensorflow2mindspore.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

本章将以LeNet5网络结构为例，结合[代码](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/convert_tf2ms_code) 来详细介绍模型权重转换方法。

首先我们需要明确训练好的TensorFlow模型转换成MindSpore能够使用的checkpoint，基本需要以下几个流程：

1. 打印TensorFlow的参数文件里面所有参数的参数名和shape，打印需要加载参数的MindSpore Cell里所有参数的参数名和shape；
2. 比较参数名和shape，构造参数映射关系；
3. 按照参数映射将TensorFlow的参数映射到MindSpore的Parameter，构成Parameter List之后保存成checkpoint；
4. 单元测试：MindSpore加载转换后的参数，固定输入，对比MindSpore与TensorFlow的结果。

## 打印参数信息

```python
# 通过TensorFlow参数文件读取模型参数的name和对应参数的shape
def tensorflow_param(ckpt_path):
    """Get TensorFlow parameter and shape"""
    tf_params = {}
    reader = tf.train.load_checkpoint(ckpt_path)
    for name in reader.get_variable_to_shape_map():
        try:
            print(name, reader.get_tensor(name).shape)
            tf_params[name] = reader.get_tensor(name)
        except Exception as e:
            pass
    return tf_params

# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    """Get MindSpore parameter and shape"""
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params
```

执行以下代码：

```python
from ms_lenet import LeNet5
tf_ckpt_path = './checkpoint_dir'
tensorflow_param(tf_ckpt_path)
print("*"*20)
network = LeNet5()
mindspore_params(network)
```

输出如下：

```text
fc3/dense/kernel (84, 1)
fc3/dense/bias (1,)
conv1/weight (5, 5, 1, 6)
fc1/dense/bias (120,)
fc1/dense/kernel (400, 120)
fc2/dense/bias (84,)
conv2/weight (5, 5, 6, 16)
fc2/dense/kernel (120, 84)
******************************
conv1.weight (6, 1, 5, 5)
conv2.weight (16, 6, 5, 5)
fc1.weight (120, 400)
fc1.bias (120,)
fc2.weight (84, 120)
fc2.bias (84,)
fc3.weight (1, 84)
fc3.bias (1,)

```

## 参数映射及checkpoint保存

通过以上参数名和shape输出进行对比，可以发现两者参数名有一定规律性可以结合网络结构进行匹配，针对参数shape可以发现卷积和全连接层的shape维度不一样，
MindSpore的卷积层中weight的shape为[out_channel, in_channel, kernel_height, kernel_weight]，而TensorFlow卷积层的weight
的shape为[kernel_height, kernel_weight, in_channel, out_channel]，MindSpore的全连接层中weight的shape为[out_channel, in_channel]，
而TensorFlow全连接层的weight的shape为[in_channel, out_channel]，所以在这里我们处理卷积和全连接层权重转换的时候需要做下转置。

```python
def tensorflow2mindspore(tf_ckpt_dir, param_mapping_dict, ms_ckpt_path):

    reader = tf.train.load_checkpoint(tf_ckpt_dir)
    new_params_list = []
    for name in param_mapping_dict:
        param_dict = {}
        parameter = reader.get_tensor(name)
        if 'conv' in name and 'weight' in name:
            # 对卷积权重进行转置
            parameter = np.transpose(parameter, axes=[3, 2, 0, 1])
        if 'fc' in name and 'kernel' in name:
            parameter = np.transpose(parameter, axes=[1, 0])
        param_dict['name'] = param_mapping_dict[name]
        param_dict['data'] = Tensor(parameter)
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, os.path.join(ms_ckpt_path, 'tf2mindspore.ckpt'))
```

因为当前网络的参数名映射非常复杂，通过参数名很难找到映射关系，所以我们需要通过一个参数映射字典。当遇到比较简单的参数名映射时，
转换方法可以参考[PyTorch模型文件转MindSpore模型文件](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/sample_code.html#%E6%A8%A1%E5%9E%8B%E9%AA%8C%E8%AF%81)的方法。

```python
params_mapping = {
    "conv1/weight":"conv1.weight",
    "conv2/weight":"conv2.weight",
    "fc1/dense/kernel":"fc1.weight",
    "fc1/dense/bias":"fc1.bias",
    "fc2/dense/kernel":"fc2.weight",
    "fc2/dense/bias":"fc2.bias",
    "fc3/dense/kernel":"fc3.weight",
    "fc3/dense/bias":"fc3.bias",
}
ms_ckpt_path='./model'
tf_ckpt_dir = './model'
tensorflow2mindspore(tf_ckpt_dir, param_mapping_dir, ms_ckpt_path)
```

执行完成后可以在相应路径下找到转换后MindSpore可以使用的模型文件。

## 单元测试

获得对应的参数文件后，我们需要对整个模型做一次单元测试，保证模型的一致性：

```python
from ms_lenet import mindspore_running
from tf_lenet import tf_running

tf_model_path = './model'
tf_outputs = tf_running(tf_model_path)
ms_outputs = mindspore_running('./tf2mindspore.ckpt')
diff = mean_relative_error(tf_outputs, ms_outputs)
print("************tensorflow outputs**************")
print(tf_outputs)
print("************mindspore outputs**************")
print(ms_outputs)
print("Diff: ", diff)
```

输出

```text
************tensorflow outputs**************
[[56.040612]
 [56.040612]
 [56.040612]
 [56.040612]
 [56.040612]
 [56.040612]
 [56.04064 ]
 [56.04064 ]]
************mindspore outputs**************
[[56.04065]
 [56.04065]
 [56.04065]
 [56.04065]
 [56.04065]
 [56.04065]
 [56.04065]
 [56.04065]]
Diff:  5.4456143e-07

```

可以看到最后的结果相差不大，基本符合预期。