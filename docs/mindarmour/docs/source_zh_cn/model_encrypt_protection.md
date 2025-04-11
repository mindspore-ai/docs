# 模型加密保护

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_zh_cn/model_encrypt_protection.md)

## 概述

MindSpore框架提供通过加密对模型文件进行保护的功能，使用对称加密算法对参数文件或推理模型进行加密，使用时直接加载密文模型完成推理或增量训练。
目前加密方案支持在Linux平台下对CheckPoint和MindIR模型文件的保护。

以下通过示例来介绍加密导出和解密加载的方法。

> 你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_encrypt_protection/encrypt_checkpoint.py>

## 安全导出CheckPoint文件

目前MindSpore支持用Callback机制在训练过程中保存模型参数，用户可以在`CheckpointConfig`对象中配置加密密钥和加密模式，并将其传入`ModelCheckpoint`来启用参数文件的加密保护。具体配置方法如下：

```python
from mindspore.train import CheckpointConfig, ModelCheckpoint

config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10, enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
ckpoint_cb = ModelCheckpoint(prefix='lenet_enc', directory=None, config=config_ck)
model.train(10, dataset, callbacks=ckpoint_cb)
```

上述代码中，通过在`CheckpointConfig`中初始化加密密钥和加密模式来启用模型加密。

- `enc_key`表示用于对称加密的密钥。

- `enc_mode`表示使用哪种加密模式。

除了上面这种保存模型参数的方法，还可以调用`save_checkpoint`接口来保存模型参数，使用方法如下：

```python
import mindspore as ms

ms.save_checkpoint(network, 'lenet_enc.ckpt', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
```

其中`enc_key`和`enc_mode`的定义同上。

## 加载密文CheckPoint文件

MindSpore提供`load_checkpoint`和`load_distributed_checkpoint`分别用于单文件和分布式场景下加载CheckPoint参数文件。以单文件场景为例，可以用如下方式加载密文CheckPoint文件：

```python
import mindspore as ms

param_dict = ms.load_checkpoint('lenet_enc.ckpt', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
```

上述代码中，通过指定`dec_key`和`dec_mode`来启用对密文文件的读取。

- `dec_key`表示用于对称解密的密钥。

- `dec_mode`表示使用哪种解密模式。

分布式场景的方式类似，在调用`load_distributed_checkpoint`时指定`dec_key`和`dec_mode`即可。

## 安全导出模型文件

MindSpore提供的`export`接口可导出MindIR、AIR、ONNX等格式的模型，在导出MindIR模型时可用如下方式启用加密保护：

```python
import mindspore as ms
input_arr = ms.Tensor(np.zeros([32, 3, 32, 32], np.float32))
ms.export(network, input_arr, file_name='lenet_enc', file_format='MINDIR', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
```

AIR、ONNX、MindIR格式支持自定义加密保护，自定义加密函数需满足如下规范：

```python
def encrypt_func(model_stream : bytes, key : bytes):
    plain_data = BytesIO()
    # 自定义加密算法
    plain_data.write(model_stream)
    return plain_data.getvalue()
```

其中，自定义加密函数的参数为二进制格式的模型（bytes）和密钥（bytes），并返回加密后的二进制序列化模型，自定义加密算法从`enc_mode`处传入。

具体用法如下：

```python
import mindspore as ms
def encrypt_func(model_stream : bytes, key : bytes):
    plain_data = BytesIO()
    # 自定义加密算法
    plain_data.write(model_stream)
    return plain_data.getvalue()

input_arr = ms.Tensor(np.zeros([32, 3, 32, 32], np.float32))
ms.export(network, input_arr, file_name='lenet_enc', file_format='MINDIR', enc_key=b'0123456789ABCDEF', enc_mode=encrypt_func)
```

## 加载密文MindIR文件

云侧使用Python编写脚本，可以用`load`接口加载MindIR模型，在加载密文MindIR时，通过指定`dec_key`和`dec_mode`对模型进行解密。

```python
import mindspore as ms
graph = ms.load('lenet_enc.mindir', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
```

如模型文件使用自定义加密导出，需使用配套自定义解密算法进行解密加载。自定义解密函数需满足如下规范：

```python
def decrypt_func(cipher_file : str, key : bytes):
    with open(cipher_file, 'rb') as f:
        plain_data = f.read()
    # 自定义解密算法
    f.close()
    return plain_data
```

其中，自定义解密函数需要两个参数：文件名（str）和解密密钥（bytes），并返回解密后的二进制模型文件。自定义解密算法从`dec_mode`处传入，解密密钥需与加密密钥保持一致。

具体用法如下：

```python
import mindspore as ms
def decrypt_func(cipher_file : str, key : bytes):
    with open(cipher_file, 'rb') as f:
        plain_data = f.read()
    # 自定义解密算法
    f.close()
    return plain_data
graph = ms.load('lenet_enc.mindir', dec_key=b'0123456789ABCDEF', dec_mode=decrypt_func)
```

> 使用自定义加解密对模型进行导出加载时，MindSpore框架不会对加解密函数的正确性进行验证，需用户自行检查算法。

对于C++脚本，MindSpore也提供了`Load`接口以加载MindIR模型，接口定义可参考[api文档](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html)：

在加载密文模型时，通过指定`dec_key`和`dec_mode`对模型进行解密。

```C++
#include "include/api/serialization.h"

namespace mindspore{
  Graph graph;
  const unsigned char[] key = "0123456789ABCDEF";
  const size_t key_len = 16;
  Key dec_key(key, key_len);
  Serialization::Load("./lenet_enc.mindir", ModelType::kMindIR, &graph, dec_key, "AES-GCM");
} // namespace mindspore
```

## 警告

在Python环境中完成加密或者解密后，需要及时清空内存中的key，参考清空方式：
调用加解密接口时，先声明一个变量key来记录密钥，比如key=b'0123456789ABCDEF'，然后传入到调用接口中去，比如save_checkpoint(network, 'lenet_enc.ckpt', enc_key=key, enc_mode='AES-GCM')。完成任务后，使用ctypes清除key：

```python
import sys
import ctypes
length = len(key)
offset = sys.getsizeof(key) - length - 1
ctypes.memset(id(key) + offset, 0, length)
```

对于运行config_ck=CheckpointConfig()传入的key，也可以用上面的方式清除，只需要把上述代码中的key换成config_ck._enc_key即可。

## 端侧模型保护

### 模型转换工具

MindSpore Lite提供的模型转换工具conveter可以将密文的mindir模型转化为明文ms模型，用户只需在调用该工具时指明密钥和解密模式即可，注意这里的密钥为十六进制表示的字符串，如前面定义的`b'0123456789ABCDEF'`对应的十六进制表示为`30313233343536373839414243444546`，Linux平台用户可以使用`xxd`工具对字节表示的密钥进行十六进制表达转换。具体调用方法如下：

```shell
./converter_lite --fmk=MINDIR --modelFile=./lenet_enc.mindir --outputFile=lenet --decryptKey=30313233343536373839414243444546 --decryptMode=AES-GCM
```

