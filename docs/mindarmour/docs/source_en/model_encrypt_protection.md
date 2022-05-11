# Model Encryption Protection

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_en/model_encrypt_protection.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

The MindSpore framework provides the symmetric encryption algorithm to encrypt the parameter files or inference models to protect the model files. When the symmetric encryption algorithm is used, the ciphertext model is directly loaded to complete inference or incremental training.
Currently, the encryption solution protects checkpoint and MindIR model files on the Linux platform.

The following uses an example to describe how to encrypt, export, decrypt, and load data.

> Download address of the complete sample code: <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_encrypt_protection/encrypt_checkpoint.py>

## Safely Exporting a Checkpoint File

Currently, MindSpore supports the use of the callback mechanism to save model parameters during training. You can configure the encryption key and encryption mode in the `CheckpointConfig` object and transfer them to the `ModelCheckpoint` to enable encryption protection for the parameter file. The configuration procedure is as follows:

```python
from mindspore import CheckpointConfig, ModelCheckpoint

config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10, enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
ckpoint_cb = ModelCheckpoint(prefix='lenet_enc', directory=None, config=config_ck)
model.train(10, dataset, callbacks=ckpoint_cb)
```

In the preceding code, the encryption key and encryption mode are initialized in `CheckpointConfig` to enable model encryption.

- `enc_key` indicates the key used for symmetric encryption.

- `enc_mode` indicates the encryption mode.

In addition to the preceding method for saving model parameters, you can also call the `save_checkpoint` API to save model parameters. The method is as follows:

```python
from mindspore import save_checkpoint

save_checkpoint(network, 'lenet_enc.ckpt', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
```

The definitions of `enc_key` and `enc_mode` are the same as those described above.

## Loading the Ciphertext Checkpoint File

MindSpore provides `load_checkpoint` and `load_distributed_checkpoint` for loading checkpoint parameter files in single-file and distributed scenarios, respectively. For example, in the single-file scenario, you can use the following method to load the ciphertext checkpoint file:

```python
from mindspore import load_checkpoint

param_dict = load_checkpoint('lenet_enc.ckpt', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
```

In the preceding code, `dec_key` and `dec_mode` are specified to enable the function of reading the ciphertext file.

- `dec_key` indicates the key used for symmetric decryption.

- `dec_mode` indicates the decryption mode.

The methods in distributed scenarios are similar. You only need to specify `dec_key` and `dec_mode` when calling `load_distributed_checkpoint`.

## Safely Exporting a MindIR File

The `export` API provided by MindSpore can be used to export models in MindIR, AIR, or ONNX format. When exporting a MindIR model, you can use the following method to enable encryption protection:

```python
from mindspore import export
input_arr = Tensor(np.zeros([32, 3, 32, 32], np.float32))
export(network, input_arr, file_name='lenet_enc', file_format='MINDIR', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
```

> Currently, the AIR and ONNX formats do not support encryption protection.

## Loading the Ciphertext MindIR File

If you write scripts using Python on the cloud, you can use the `load` API to load the MindIR model. When loading the ciphertext MindIR, you can specify `dec_key` and `dec_mode` to decrypt the model.

```python
from mindspore import load
graph = load('lenet_enc.mindir', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
```

For C++ scripts, MindSpore also provides the `Load` API to load MindIR models. For details about the API definition, see [MindSpore API](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html).

When loading a ciphertext model, you can specify `dec_key` and `dec_mode` to decrypt the model.

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

## Warning

After completing encryption or decryption in Python environment, you need to empty the key in memory in time. Refer to the emptying method:
When calling the encryption or decryption interface, first declare a variable 'key' to record the key, such as key=b'0123456789ABCDEF', and then pass it to the calling interface, such as save_checkpoint(network, 'lenet_enc.ckpt', enc_key=key, enc_mode='AES-GCM'). After completing the task, use ctypes to empty the key:

```python
import sys
import ctypes
length = len(key)
offset = sys.getsizeof(key) - length - 1
ctypes.memset(id(key) + offset, 0, length)
```

For the key passed to CheckpointConfig() when config_ck=CheckpointConfig() is run, you can also empty it as the method above by replacing 'key' in the code with 'config_ck._enc_key'.

## On-Device Model Protection

### Model Converter

The model converter provided by MindSpore Lite can convert a ciphertext MindIR model into a plaintext MS model. You only need to specify the key and decryption mode when calling this tool. Note that the key is a hexadecimal character string, for example, the hexadecimal string corresponding to `b'0123456789ABCDEF` is `30313233343536373839414243444546`. On the Linux platform, you can use the `xxd` tool to convert the key represented by bytes to a hexadecimal string. The call method is as follows:

```shell
./converter_tools --fmk=MINDIR --modelFile=./lenet_enc.mindir --outputFile=lenet --decryptKey=30313233343536373839414243444546 --decryptMode=AES-GCM
```
