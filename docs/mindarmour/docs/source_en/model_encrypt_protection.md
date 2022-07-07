# Model Encryption Protection

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/mindarmour/docs/source_en/model_encrypt_protection.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

The MindSpore framework provides the symmetric encryption algorithm to encrypt the parameter files or inference models to protect the model files. When the symmetric encryption algorithm is used, the ciphertext model is directly loaded to complete inference or incremental training.
Currently, the encryption solution protects checkpoint and MindIR model files on the Linux platform.

The following uses an example to describe how to encrypt, export, decrypt, and load data.

> Download address of the complete sample code: <https://gitee.com/mindspore/docs/blob/r1.8/docs/sample_code/model_encrypt_protection/encrypt_checkpoint.py>

## Safely Exporting a Checkpoint File

Currently, MindSpore supports the use of the callback mechanism to save model parameters during training. You can configure the encryption key and encryption mode in the `CheckpointConfig` object and transfer them to the `ModelCheckpoint` to enable encryption protection for the parameter file. The configuration procedure is as follows:

```python
import mindspore as ms

config_ck = ms.CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10, enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
ckpoint_cb = ms.ModelCheckpoint(prefix='lenet_enc', directory=None, config=config_ck)
model.train(10, dataset, callbacks=ckpoint_cb)
```

In the preceding code, the encryption key and encryption mode are initialized in `CheckpointConfig` to enable model encryption.

- `enc_key` indicates the key used for symmetric encryption.

- `enc_mode` indicates the encryption mode.

In addition to the preceding method for saving model parameters, you can also call the `save_checkpoint` API to save model parameters. The method is as follows:

```python
import mindspore as ms

ms.save_checkpoint(network, 'lenet_enc.ckpt', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
```

The definitions of `enc_key` and `enc_mode` are the same as those described above.

## Loading the Ciphertext Checkpoint File

MindSpore provides `load_checkpoint` and `load_distributed_checkpoint` for loading checkpoint parameter files in single-file and distributed scenarios, respectively. For example, in the single-file scenario, you can use the following method to load the ciphertext checkpoint file:

```python
import mindspore as ms

param_dict = ms.load_checkpoint('lenet_enc.ckpt', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
```

In the preceding code, `dec_key` and `dec_mode` are specified to enable the function of reading the ciphertext file.

- `dec_key` indicates the key used for symmetric decryption.

- `dec_mode` indicates the decryption mode.

The methods in distributed scenarios are similar. You only need to specify `dec_key` and `dec_mode` when calling `load_distributed_checkpoint`.

## Safely Exporting a Model File

The `export` API provided by MindSpore can be used to export models in MindIR, AIR, or ONNX format. When exporting a MindIR model, you can use the following method to enable encryption protection:

```python
import mindspore as ms
input_arr = ms.Tensor(np.zeros([32, 3, 32, 32], np.float32))
ms.export(network, input_arr, file_name='lenet_enc', file_format='MINDIR', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
```

MindIR, AIR, or ONNX formats support user-customized encryption protection. The encryption method must follow the format below:

```python
def encrypt_func(model_stream : bytes, key : bytes):
    plain_data = BytesIO()
    # customized encryption algorithm
    plain_data.write(model_stream)
    return plain_data.getvalue()
```

The parameters for customized encryption are model stream (bytes) and encryption key (bytes). The encryption method must return the encrypted model stream in bytes too. The customized encryption method is passed from the parameter `enc_mode`.

You can use the following method to enable customized encryption protection:

```python
import mindspore as ms
def encrypt_func(model_stream : bytes, key : bytes):
    plain_data = BytesIO()
    # customized encryption algorithm
    plain_data.write(model_stream)
    return plain_data.getvalue()

input_arr = ms.Tensor(np.zeros([32, 3, 32, 32], np.float32))
ms.export(network, input_arr, file_name='lenet_enc', file_format='MINDIR', enc_key=b'0123456789ABCDEF', enc_mode=encrypt_func)
```

## Loading the Ciphertext MindIR File

If you write scripts using Python on the cloud, you can use the `load` API to load the MindIR model. When loading the ciphertext MindIR, you can specify `dec_key` and `dec_mode` to decrypt the model.

```python
import mindspore as ms
graph = ms.load('lenet_enc.mindir', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
```

If the model is exported with customized encryption method, you should load the cipher file with customized decryption method. The decryption method must follow the format below:

```python
def decrypt_func(cipher_file : str, key : bytes):
    with open(cipher_file, 'rb') as f:
        plain_data = f.read()
    # customized decryption algorithm
    f.close()
    return plain_data
```

The parameters for customized decryption are cipher file name (str) and decryption key (bytes). The decryption method must return the decrypted model stream in bytes. The customized decryption method is passed from the parameter `dec_mode`. The decrpytion key and the encryption key should be the same.

You can use the following method to enable loading the customized-decrypted model:

```python
import mindspore as ms
def decrypt_func(cipher_file : str, key : bytes):
    with open(cipher_file, 'rb') as f:
        plain_data = f.read()
    # customized decryption algorithm
    f.close()
    return plain_data
graph = ms.load('lenet_enc.mindir', dec_key=b'0123456789ABCDEF', dec_mode=decrypt_func)
```

> When using the customized encryption-decryption to export and load the model, the MindSpore framework would not check the correctness of encryption/decryption algorithm. The user should guarantee the correctness of such algorithms.

For C++ scripts, MindSpore also provides the `Load` API to load MindIR models. For details about the API definition, see [MindSpore API](https://www.mindspore.cn/lite/api/en/r1.8/api_cpp/mindspore.html).

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

For the key passed to ms.CheckpointConfig() when config_ck=ms.CheckpointConfig() is run, you can also empty it as the method above by replacing 'key' in the code with 'config_ck._enc_key'.

## On-Device Model Protection

### Model Converter

The model converter provided by MindSpore Lite can convert a ciphertext MindIR model into a plaintext MS model. You only need to specify the key and decryption mode when calling this tool. Note that the key is a hexadecimal character string, for example, the hexadecimal string corresponding to `b'0123456789ABCDEF` is `30313233343536373839414243444546`. On the Linux platform, you can use the `xxd` tool to convert the key represented by bytes to a hexadecimal string. The call method is as follows:

```shell
./converter_tools --fmk=MINDIR --modelFile=./lenet_enc.mindir --outputFile=lenet --decryptKey=30313233343536373839414243444546 --decryptMode=AES-GCM
```
