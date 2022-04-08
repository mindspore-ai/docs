# 第三方框架迁移使用

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/faq/source_zh_cn/usage_migrate_3rd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: 请问想加载PyTorch预训练好的模型用于MindSpore模型finetune有什么方法？**</font>

A: 需要把PyTorch和MindSpore的参数进行一一对应，因为网络定义的灵活性，所以没办法提供统一的转化脚本。

一般情况下，CheckPoint文件中保存的就是参数名和参数值，调用相应框架的读取接口后，获取到参数名和数值后，按照MindSpore格式，构建出对象，就可以直接调用MindSpore接口保存成MindSpore格式的CheckPoint文件了。

其中主要的工作量为对比不同框架间的parameter名称，做到两个框架的网络中所有parameter name一一对应(可以使用一个map进行映射)，下面代码的逻辑转化parameter格式，不包括对应parameter name。

```python
import torch
from mindspore import Tensor, save_checkpoint

def pytorch2mindspore(default_file = 'torch_resnet.pth'):
    # read pth file
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

<font size=3>**Q: 怎么将PyTorch的`dataset`转换成MindSpore的`dataset`？**</font>

A: MindSpore和PyTorch的自定义数据集逻辑是比较类似的，需要用户先定义一个自己的`dataset`类，该类负责定义`__init__`，`__getitem__`,`__len__`来读取自己的数据集，然后将该类实例化为一个对象（如: `dataset/dataset_generator`），最后将这个实例化对象传入`GeneratorDataset`(mindspore用法)/`DataLoader`(pytorch用法)，至此即可以完成自定义数据集加载了。而MindSpore在`GeneratorDataset`的基础上提供了进一步的`map`->`batch`操作，可以很方便的让用户在`map`内添加一些其他的自定义操作，并将其`batch`起来。
对应的MindSpore的自定义数据集加载如下:

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

<font size=3>**Q: 其他框架的脚本或者模型怎么迁移到MindSpore？**</font>

A: 关于脚本或者模型迁移，可以查询MindSpore官网中关于[迁移脚本](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/migration_script.html)的介绍。

<br/>

<font size=3>**Q: MindConverter转换TensorFlow脚本报错提示`terminate called after throwing an instance of 'std::system_error', what(): Resource temporarily unavailable, Aborted (core dumped)`**</font>

A: 该问题由TensorFlow导致。脚本转换时，需要通过TensorFlow库加载TensorFlow的模型文件，此时TensorFlow会申请相关资源进行初始化，若申请资源失败（可能由于系统进程数超过Linux最大进程数限制），TensorFlow C/C++层会出现Core Dumped问题。详细信息请参考TensorFlow官方ISSUE，如下ISSUE仅供参考: [TF ISSUE 14885](https://github.com/tensorflow/tensorflow/issues/14885), [TF ISSUE 37449](https://github.com/tensorflow/tensorflow/issues/37449)

<br/>

<font size=3>**Q: MindConverter是否可以在ARM平台运行？**</font>

A: MindConverter同时支持X86、ARM平台，若在ARM平台运行需要用户自行安装模型所需的依赖包和运行环境。

<br/>

<font size=3>**Q: 为什么使用MindConverter进行模型转换需要很长时间（超过十分钟），而模型并不大？**</font>

A: MindConverter进行转换时，需要使用Protobuf对模型文件进行反序列化，请确保Python环境中安装的Protobuf采用C++后端实现，检查方法如下，若输出为Python，则需要安装采用C++实现的Python Protobuf（下载Protobuf源码并进入源码中的python子目录，使用python setup.py install --cpp_implementation进行安装）；若输出为cpp，转换过程仍耗时较长，请在转换前使用添加环境变量`export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`。

```python
from google.protobuf.internal import api_implementation

print(api_implementation.Type())
```

<br/>

<font size=3>**Q: 使用.pb文件进行转换时，已确定`model_file`，`shape`，`input_nodes`，`output_nodes`均无误，并且环境中的依赖库已经正常安装，但是仍然报异常代码1000001，可能是什么原因？**</font>

A: 请检查生成该.pb文件所使用的TensorFlow版本不高于用于转换时安装的TensorFlow版本，避免由于旧版本TensorFlow无法解析新版本生成的.pb文件，而导致的模型文件解析失败。

<br/>

<font size=3>**Q: 出现报错信息`[ERROR] MINDCONVERTER: [BaseConverterError] code: 0000000, msg: {python_home}/lib/libgomp.so.1: cannot allocate memory in static TLS block`时，应该怎么处理？**</font>

A: 该问题通常是由于环境变量导入不正确导致的。建议用户设置`export LD_PRELOAD={python_home}/lib/libgomp.so.1.0.0`这一环境变量，然后重新尝试进行转换。
