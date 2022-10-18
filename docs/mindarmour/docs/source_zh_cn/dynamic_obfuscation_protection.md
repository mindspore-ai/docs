# 模型动态混淆

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_zh_cn/dynamic_obfuscation_protection.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore框架提供通过动态混淆对MindIR模型进行保护的功能，混淆后的模型结构与原模型不同，即使敌手窃取到了模型，也不知道原模型的网络结构。
模型拥有者可以使用MindSpore原生的 `load()` 接口加载混淆模型进行推理，模型在推理过程中也是混淆态的，保证了模型运行时的安全性。

目前动态混淆支持在Linux平台下通过使用MindSpore原生Python接口 `export()` 导出混淆模型或者使用Python接口 `obfuscate_model()` 导出混淆模型。

以下通过示例来介绍导出和加载混淆模型的方法。

> 你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dynamic_obfuscation/test_dynamic_obfuscation.py>

## 使用 `export()` 接口导出混淆模型及部署推理

`export()` 是MindSpore提供的模型导出接口，该接口可以将 `nn.Cell` 类的网络导出成多种格式的模型文件：

```python
mindspore.export(net, *inputs, file_name, file_format="AIR", **kwargs)
```

如果要使用这个接口导出混淆后的模型，需要设置动态混淆的字典参数 `obf_config` ，然后作为kwargs传入。动态混淆提供了两种模式来保护模型，分别是password模式和customized function模式。
下面分别介绍在这两种模式下如何导出混淆模型和加载混淆模型。

### Password模式－导出混淆模型

1. 准备实验网络，我们如下构建了ObfuscateNet进行实验：

    ```python
    import numpy as np
    import mindspore as ms
    import mindspore.ops as ops
    import mindspore.nn as nn
    from mindspore.common.initializer import TruncatedNormal
    ms.context.set_context(mode=ms.context.GRAPH_MODE)

    def weight_variable():
        return TruncatedNormal(0.02)

    def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
        weight = weight_variable()
        return nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         weight_init=weight, has_bias=False, pad_mode="valid")

    def fc_with_initialize(input_channels, out_channels):
        weight = weight_variable()
        bias = weight_variable()
        return nn.Dense(input_channels, out_channels, weight, bias, has_bias=False)

    class ObfuscateNet(nn.Cell):
        def __init__(self):
            super(ObfuscateNet, self).__init__()
            self.batch_size = 32
            self.conv1 = conv(1, 6, 5)
            self.conv2 = conv(6, 16, 5)
            self.matmul = ops.MatMul()
            self.matmul_weight1 = ms.Tensor(np.random.random((16 * 5 * 5, 120)).astype(np.float32))
            self.matmul_weight2 = ms.Tensor(np.random.random((120, 84)).astype(np.float32))
            self.matmul_weight3 = ms.Tensor(np.random.random((84, 10)).astype(np.float32))
            self.relu = nn.ReLU()
            self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()

        def construct(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.flatten(x)
            x = self.matmul(x, self.matmul_weight1)
            x = self.relu(x)
            x = self.matmul(x, self.matmul_weight2)
            x = self.relu(x)
            x = self.matmul(x, self.matmul_weight3)
            return x
    ```

2. 设置混淆参数字典：

    ```python
    obf_config = {"obf_ratio": 0.8, "obf_password": 3423}
    ```

    如上所示，参数 `obf_ratio` 表示混淆比例，代表混淆模型中的混淆节点占全部模型节点数量的比例，取值可以是浮点数或者字符串。如果是浮点数，其取值范围是(0, 1]，如果是字符串，合法取值是 `'small'` 、 `'medium'` 或 `'large'` 。
    `obf_password` 是混淆密码，合法值是大于0、小于int64_max（9223372036854775807）的整数。

3. 导出混淆模型：

    ```python
    net = ObfuscateNet()
    input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    obf_config = {"obf_ratio": 0.8, "obf_password": 3423}
    ms.export(net, input_tensor, file_name="obf_net", file_format="MINDIR", obf_config=obf_config)
    ```

    完成上述步骤后，就可以得到一个混淆后的MindIR模型（目前只支持导出MindIR格式的混淆模型）。

### Password模式－加载混淆模型

使用MindSpore的 `load()` 接口和 `nn.GraphCell()` 接口可以加载混淆模型进行推理。注意，在调用 `nn.GraphCell()` 接口时，需要把正确的 `obf_password` 传入，否则得到的推理结果不正确。

```python
obf_graph = ms.load("obf_net.mindir")
obf_net = nn.GraphCell(obf_graph, obf_password=3423)
right_password_result = obf_net(input_tensor).asnumpy()
print(right_password_result)
# [[743.6489  844.62427 716.82104 735.7657  802.5662  833.0927  861.00336 769.6415  857.3915  765.9037 ]]
```

为了验证推理结果是否正确，可以如下得到原模型的推理结果，并将 `right_password_result` 与之比较:

```python
original_predict_result = net(input_tensor).asnumpy()
print(original_predict_result)
# [[743.6489  844.62427 716.82104 735.7657  802.5662  833.0927  861.00336 769.6415  857.3915  765.9037 ]]
print(np.all(original_predict_result == right_password_result))
# True
```

比较结果表明输入正确password得到的推理结果与原模型的推理结果完全一致，且精度无损。

### Customized function模式（只支持CPU）－导出混淆模型

除了password模式，动态混淆还提供自定义函数模式(customized function mode)。自定义函数模式相比于password模式安全性更高，但配置方式相对复杂。
自定义函数模式需要用户定义一个Python函数，它需要满足这些要求：１、入参数量为2；２、对于任意输入（输入都来自于模型中任意一层在推理过程中的输出值），该函数的输出值恒为True或者False。例如：

```python
def my_func(x1, x2):
    if abs(x1) + abs(x2) < 0:
        return True
    return False
```

注意：感兴趣的用户可以搜索**不透明谓词表达式**，来构造满足上述要求的函数，后续我们也会增加不透明谓词表达式的构造教程。

准备好 `cutomized_func` 之后，我们就可以如下导出混淆模型了：

```python
obf_config = {"obf_ratio": 0.8, "customized_func": my_func}
net = ObfuscateNet()
ms.export(net, input_tensor, file_name="obf_net", file_format="MINDIR", obf_config=obf_config)
```

### Customized function模式－加载混淆模型

使用MindSpore的 `load()` 接口和 `nn.GraphCell()` 接口可以加载混淆模型进行推理。注意，在调用 `load()` 接口时，需要把正确的 `customized_func` （函数名和函数体都需要和混淆时设置的函数保持一致）传入，否则得到的推理结果不正确。

```python
obf_graph = ms.load("obf_net.mindir", obf_func=my_func)
obf_net = nn.GraphCell(obf_graph)
right_func_result = obf_net(input_tensor).asnumpy()
```

## 使用 `obfuscate_model()` 接口导出混淆模型及部署推理

`obfuscate_model()` 是专门用于模型混淆的接口，该接口可以将未混淆的MindIR模型转换为混淆后的MindIR模型：

```python
ms.obfuscate_model(obf_config, **kwargs)
```

`obfuscate_model()` 也提供了password模式和customized function模式。
下面分别介绍在这两种模式下如何导出混淆模型和加载混淆模型。

### Password模式－导出混淆模型

1. 设置混淆参数字典：

    ```python
    input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    obf_config = {"original_model_path": "net.mindir", "save_model_path": "./obf_net",
                  "model_inputs": [input_tensor], "obf_ratio": 0.8, "obf_password": 3423}
    ```

    如上所示，参数 `original_model_path` 指的是待混淆的模型路径； `save_model_path` 指的是混淆模型的输出保存路径； `model_inputs` 指的是模型输入的Tensor，Tensor的值可以是随机数，与 `export()` 的inputs类似。
    参数 `obf_ratio` 和 `obf_password` 与介绍 `export()` 接口时相同。

2. 导出混淆模型：

    ```python
    ms.obfuscate_model(obf_config)
    ```

    完成上述步骤后，就可以得到一个混淆后的MindIR模型。

### Password模式－加载混淆模型

使用MindSpore的 `load()` 接口和 `nn.GraphCell()` 接口可以加载混淆模型进行推理。注意，在调用 `nn.GraphCell()` 接口时，需要把正确的 `obf_password` 传入，否则得到的推理结果不正确。

```python
obf_graph = ms.load("obf_net.mindir")
obf_net = nn.GraphCell(obf_graph, obf_password=3423)
right_password_result = obf_net(input_tensor).asnumpy()
```

为了验证推理结果是否正确，可以如下得到原模型的推理结果，并将 `right_password_result` 与之比较:

```python
original_graph = ms.load("net.mindir")
original_net = nn.GraphCell(original_graph)
original_predict_result = original_net(input_tensor).asnumpy()
print(np.all(original_predict_result == right_password_result))
# True
```

### Customized function模式（只支持CPU）－导出混淆模型

和 `export` 接口相同，用户需要准备一个自定义函数：

```python
def my_func(x1, x2):
    if abs(x1) + abs(x2) < 0:
        return True
    return False
```

然后如下导出混淆模型：

```python
net = ObfuscateNet()
input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
ms.export(net, input_tensor, file_name="net", file_format="MINDIR")
obf_config = {"original_model_path": "net.mindir", "save_model_path": "./obf_net",
              "model_inputs": [input_tensor], "obf_ratio": 0.8, "customized_func": my_func}
ms.obfuscate_model(obf_config)
```

### Customized function模式－加载混淆模型

使用MindSpore的 `load()` 接口和 `nn.GraphCell()` 接口可以加载混淆模型进行推理。注意，在调用 `load()` 接口时，需要把正确的 `customized_func` （函数名和函数体都需要和混淆时设置的函数保持一致）传入，否则得到的推理结果不正确。

```python
obf_graph = ms.load("obf_net.mindir", obf_func=my_func)
obf_net = nn.GraphCell(obf_graph)
right_func_result = obf_net(input_tensor).asnumpy()
```
