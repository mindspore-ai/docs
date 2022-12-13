# Dynamic Model Obfuscation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_en/dynamic_obfuscation_protection.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

The MindSpore framework can protect a MindIR model through dynamic obfuscation. The structure of an obfuscated model is different from that of the original model. Therefore, even if someone steals the model, they do not know the network structure of the original model.
The model owner can use the native MindSpore API `load()` to load the obfuscated model for inference. The model is also in the obfuscation state during inference, ensuring the security of the model at runtime.

Currently, obfuscated models can be exported from the Linux platform using the native Python API `export()` of MindSpore or the Python API `obfuscate_model()`.

The following uses an example to describe how to export and load an obfuscated model.

> You can download the complete sample code at <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dynamic_obfuscation/test_dynamic_obfuscation.py>.

## Using `export()` to Export Obfuscated Models and Deploying Inference

`export()` is a model export API provided by MindSpore. This API can export `nn.Cell` networks to model files in multiple formats.

```python
mindspore.export(net, *inputs, file_name, file_format="MINDIR", **kwargs)
```

To use this API to export an obfuscated model, you need to set the dictionary parameter `obf_config` for dynamic obfuscation and input it as `kwargs`. Dynamic obfuscation provides two modes to protect models: password and customized function.
The following describes how to export and load obfuscated models in the two modes.

### Exporting an Obfuscated Model in Password Mode

1. Prepare a test network ObfuscateNet:

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

2. Set the obfuscation dictionary parameter.

    ```python
    obf_config = {"obf_ratio": 0.8, "obf_password": 3423}
    ```

    As shown in the preceding information, `obf_ratio` indicates the obfuscation ratio, that is, the ratio of obfuscated nodes in the obfuscated model to all model nodes. The value can be a floating point number or a character string. If the value is a floating point number, the value range is (0,1]. If the value is a character string, the valid value is `'small'`, `'medium'`, or `'large'`.
    `obf_password` indicates the obfuscation password. The valid value is an integer greater than 0 and less than or equal to int64_max (9223372036854775807).

3. Export the obfuscated model.

    ```python
    net = ObfuscateNet()
    input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    obf_config = {"obf_ratio": 0.8, "obf_password": 3423}
    ms.export(net, input_tensor, file_name="obf_net", file_format="MINDIR", obf_config=obf_config)
    ```

    After the preceding steps are complete, an obfuscated MindIR model is obtained. (Currently, only obfuscated models in MindIR format can be exported.)

### Loading the Obfuscated Model in Password Mode

The `load()` and `nn.GraphCell()` APIs of MindSpore can be used to load obfuscated models for inference. Ensure that the input `obf_password` is correct when calling the `nn.GraphCell()` API. Otherwise, the obtained inference result is incorrect.

```python
obf_graph = ms.load("obf_net.mindir")
obf_net = nn.GraphCell(obf_graph, obf_password=3423)
right_password_result = obf_net(input_tensor).asnumpy()
print(right_password_result)
# [[743.6489  844.62427 716.82104 735.7657  802.5662  833.0927  861.00336 769.6415  857.3915  765.9037 ]]
```

To check whether the inference result is correct, you can obtain the inference result of the original model and compare `right_password_result` with the inference result as follows:

```python
original_predict_result = net(input_tensor).asnumpy()
print(original_predict_result)
# [[743.6489  844.62427 716.82104 735.7657  802.5662  833.0927  861.00336 769.6415  857.3915  765.9037 ]]
print(np.all(original_predict_result == right_password_result))
# True
```

The comparison result shows that the inference result obtained by entering the correct password is the same as that of the original model, and the accuracy is lossless.

### Exporting Obfuscated Models in Customized Function Mode (Only for the CPU Hardware Environment)

In addition to the password mode, dynamic obfuscation also provides the customized function mode. Compared with the password mode, the customized function mode is more secure, but the configuration is more complex.
In customized function mode, you need to define a Python function that meets the following requirements: 1. There are 2 input parameters. 2. For any input (which comes from the output of any model layer during inference), the output value of this function is always True or False. Example:

```python
def my_func(x1, x2):
    if abs(x1) + abs(x2) < 0:
        return True
    return False
```

Note: You can search for **Opaque predicate expression** to construct functions that meet the preceding requirements. For example:

```python
def opaque_predicate(x, y):
    if 7*y^2 - 1 == x^2:
        return True
    return False
```

If x and y are integers, opaque_predicate(x, y) is always False. You can refer to [manufacturing cheap, resilient, and stealthy opaque constructs](https://dl.acm.org/doi/epdf/10.1145/268946.268962) or other literature to learn more about opaque predicates.

After the `customized_func` is ready, you can export the obfuscated model as follows:

```python
obf_config = {"obf_ratio": 0.8, "customized_func": my_func}
net = ObfuscateNet()
ms.export(net, input_tensor, file_name="obf_net", file_format="MINDIR", obf_config=obf_config)
```

### Loading the Obfuscated Model in Customized Function Mode

The `load()` and `nn.GraphCell()` APIs of MindSpore can be used to load obfuscated models for inference. Ensure that the input `customized_func` must be correct (the function name and body must be the same as those set during obfuscation) when calling the `load()` API. Otherwise, the obtained inference result is incorrect.

```python
obf_graph = ms.load("obf_net.mindir", obf_func=my_func)
obf_net = nn.GraphCell(obf_graph)
right_func_result = obf_net(input_tensor).asnumpy()
```

## Using `obfuscate_model()` to Export Obfuscated Models and Deploying Inference

`obfuscate_model()` is an API dedicated to MindIR model obfuscation.

```python
ms.obfuscate_model(obf_config, **kwargs)
```

`obfuscate_model()` also provides the password and customized function modes.
The following describes how to export and load obfuscated models in the two modes.

### Exporting an Obfuscated Model in Password Mode

1. Set the obfuscation dictionary parameter.

    ```python
    input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    obf_config = {"original_model_path": "net.mindir", "save_model_path": "./obf_net",
                  "model_inputs": [input_tensor], "obf_ratio": 0.8, "obf_password": 3423}
    ```

    As shown in the preceding information, `original_model_path` indicates the path of a model to be obfuscated, `save_model_path` indicates the path for storing the output of the obfuscated model, and `model_inputs` indicates the tensor input to the model. The tensor value can be a random number, which is similar to the `inputs` of `export()`.
    The `obf_ratio` and `obf_password` parameters are the same as those of the `export()` API.

2. Export the obfuscated model.

    ```python
    ms.obfuscate_model(obf_config)
    ```

    After the preceding steps are complete, an obfuscated MindIR model is obtained.

### Loading the Obfuscated Model in Password Mode

The `load()` and `nn.GraphCell()` APIs of MindSpore can be used to load obfuscated models for inference. Ensure that the input `obf_password` is correct when calling the `nn.GraphCell()` API. Otherwise, the obtained inference result is incorrect.

```python
obf_graph = ms.load("obf_net.mindir")
obf_net = nn.GraphCell(obf_graph, obf_password=3423)
right_password_result = obf_net(input_tensor).asnumpy()
```

To check whether the inference result is correct, you can obtain the inference result of the original model and compare `right_password_result` with the inference result as follows:

```python
original_graph = ms.load("net.mindir")
original_net = nn.GraphCell(original_graph)
original_predict_result = original_net(input_tensor).asnumpy()
print(np.all(original_predict_result == right_password_result))
# True
```

### Exporting Obfuscated Models in Customized Function Mode (Only for the CPU Hardware Environment)

Similar to the `export` API, you need to prepare a customized function.

```python
def my_func(x1, x2):
    if abs(x1) + abs(x2) < 0:
        return True
    return False
```

Export the obfuscated model as follows:

```python
net = ObfuscateNet()
input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
ms.export(net, input_tensor, file_name="net", file_format="MINDIR")
obf_config = {"original_model_path": "net.mindir", "save_model_path": "./obf_net",
              "model_inputs": [input_tensor], "obf_ratio": 0.8, "customized_func": my_func}
ms.obfuscate_model(obf_config)
```

### Loading the Obfuscated Model in Customized Function Mode

The `load()` and `nn.GraphCell()` APIs of MindSpore can be used to load obfuscated models for inference. Ensure that the input `customized_func` must be correct (the function name and body must be the same as those set during obfuscation) when calling the `load()` API. Otherwise, the obtained inference result is incorrect.

```python
obf_graph = ms.load("obf_net.mindir", obf_func=my_func)
obf_net = nn.GraphCell(obf_graph)
right_func_result = obf_net(input_tensor).asnumpy()
```

Note: Dynamic obfuscation currently does not support dynamic shape input, so `inputs` / `model_inputs` cannot be in the form of dynamic shapes (some dimensions of Tensors are None) when calling `export()` / `obfuscate_model()`. For example, `inputs=Tensor(shape=[1, 1, None, None], dtype=ms.float32)` is not allowed.