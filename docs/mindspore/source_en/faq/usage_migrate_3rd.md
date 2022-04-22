# Migration from a Third-party Framework

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/usage_migrate_3rd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: How do I load a pre-trained PyTorch model for fine-tuning on MindSpore?**</font>

A: Map parameters of PyTorch and MindSpore one by one. No unified conversion script is provided due to flexible network definitions.

In general, the parameters names and parameters values are saved in the CheckPoint file. After invoking the loading interface of the corresponding framework and obtaining the parameter names and values, construct the object according to the MindSpore format, and then you can directly invoke the MindSpore interface to save as CheckPoint files in the MindSpore format.

The main work is to compare the parameter names between different frameworks, so that all parameter names in the network of the two frameworks correspond to each other (a map can be used for mapping). The logic of the following code is transforming the parameter format, excluding the corresponding parameter name.

```python
import torch
from mindspore import Tensor, save_checkpoint

def pytorch2mindspore(default_file = 'torch_resnet.pth'):
    """read pth file"""
    par_dict = torch.load(default_file)['state_dict']
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        params_list.append(param_dict)
    save_checkpoint(params_list,  'ms_resnet.ckpt')
```

<br/>

<font size=3>**Q: How do I convert a PyTorch `dataset` to a MindSpore `dataset`?**</font>

A: The customized dataset logic of MindSpore is similar to that of PyTorch. You need to define a `dataset` class containing `__init__`, `__getitem__`, and `__len__` to read your dataset, instantiate the class into an object (for example, `dataset/dataset_generator`), and transfer the instantiated object to `GeneratorDataset` (on MindSpore) or `DataLoader` (on PyTorch). Then, you are ready to load the customized dataset. MindSpore provides further `map`->`batch` operations based on `GeneratorDataset`. Users can easily add other customized operations to `map` and start `batch`.
The customized dataset of MindSpore is loaded as follows:

```python
# 1 Data enhancement,shuffle,sampler.
class Mydata:
    def __init__(self):
        np.random.seed(58)
        self.__data = np.random.sample((5, 2))
        self.__label = np.random.sample((5, 1))
    def __getitem__(self, index):
        return (self.__data[index], self.__label[index])
    def __len__(self):
        return len(self.__data)
dataset_generator = Mydata()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
# 2 Customized data enhancement
dataset = dataset.map(operations=pyFunc, {other_params})
# 3 batch
dataset = dataset.batch(batch_size, drop_remainder=True)
```

<br/>

<font size=3>**Q: How do I migrate scripts or models of other frameworks to MindSpore?**</font>

A: For details about script or model migration, please visit the [Migration Script](https://www.mindspore.cn/docs/en/master/migration_guide/migration_script.html) in MindSpore official website.

<br/>

<font size=3>**Q: MindConverter converts TensorFlow script error prompt`terminate called after throwing an instance of 'std::system_error', what(): Resource temporarily unavailable, Aborted (core dumped)`**</font>

A: This problem is caused by TensorFlow. During script conversion, you need to load the TensorFlow model file through the TensorFlow library. At this time, TensorFlow will apply for relevant resources for initialization. If the resource application fails (maybe because the number of system processes exceeds the maximum number of Linux processes), the TensorFlow C/C++ layer will appear Core Dumped problem. For more information, please refer to the official ISSUE of TensorFlow. The following ISSUE is for reference only: [TF ISSUE 14885](https://github.com/tensorflow/tensorflow/issues/14885), [TF ISSUE 37449](https://github.com/tensorflow/tensorflow/issues/37449).

<br/>

<font size=3>**Q: Can MindConverter run on ARM platform?**</font>

A: MindConverter supports both x86 and ARM platforms. Please ensure all required dependencies and environments have been installed in the ARM platform.

<br/>

<font size=3>**Q: Why does the conversion process take a lot of time (more than 10 minutes) by using MindConverter, but the model is not so large?**</font>

A: When converting, MindConverter needs to use Protobuf to deserialize the model file. Please make sure that the Protobuf installed in Python environment is implemented by C++ backend. The validation method is as follows. If the output is "python", you need to install Python Protobuf implemented by C++ (download the Protobuf source code, enter the "python" subdirectory in the source code, and use python setup.py install --cpp_implementation to install). If the output is cpp and the conversion process still takes a long time, please add environment variable `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp` before conversion.

```python
from google.protobuf.internal import api_implementation
print(api_implementation.Type())
```

<br/>

<font size=3>**Q: While converting .pb file to MindSpore script, what may be the cause of error code 1000001 with ensuring `model_file`, `shape`, `iput_nodes` and `output_nodes` set right and third party requirements installed correctly?**</font>

A: Make sure that the TensorFlow version to generate .pb file is no higher than that to convert .pb file, and avoid the conflict which caused by using low version TensorFlow to parse .pb file generated by the high version.

<br/>

<font size=3>**Q: What should I do to deal with an error `[ERROR] MINDCONVERTER: [BaseConverterError] code: 0000000, msg: {python_home}/lib/libgomp.so.1: cannot allocate memory in static TLS block`?**</font>

A: In most cases, the problem is caused by environment variable exported incorrectly. Please set `export LD_PRELOAD={python_home}/lib/libgomp.so.1.0.0`, then try to run MindConverter again.
