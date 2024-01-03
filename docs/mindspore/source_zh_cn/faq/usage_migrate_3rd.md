# 第三方框架迁移使用

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/faq/usage_migrate_3rd.md)

<font size=3>**Q: 请问想加载PyTorch预训练好的模型用于MindSpore模型finetune有什么方法？**</font>

A: 需要把PyTorch和MindSpore的参数进行一一对应，因为网络定义的灵活性，所以没办法提供统一的转化脚本。

一般情况下，CheckPoint文件中保存的就是参数名和参数值，调用相应框架的读取接口后，获取到参数名和数值后，按照MindSpore格式，构建出对象，就可以直接调用MindSpore接口保存成MindSpore格式的CheckPoint文件了。

其中主要的工作量为对比不同框架间的parameter名称，做到两个框架的网络中所有parameter name一一对应(可以使用一个map进行映射)，下面代码的逻辑转化parameter格式，不包括对应parameter name。

```python
import torch
import mindspore as ms

def pytorch2mindspore(default_file = 'torch_resnet.pth'):
    # read pth file
    par_dict = torch.load(default_file)['state_dict']
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        param_dict['name'] = name
        param_dict['data'] = ms.Tensor(parameter.numpy())
        params_list.append(param_dict)
    ms.save_checkpoint(params_list,  'ms_resnet.ckpt')
```

<br/>

<font size=3>**Q: 怎么将PyTorch的`dataset`转换成MindSpore的`dataset`？**</font>

A: MindSpore和PyTorch的自定义数据集逻辑是比较类似的，需要用户先定义一个自己的`dataset`类，该类负责定义`__init__`、`__getitem__`、`__len__`来读取自己的数据集，然后将该类实例化为一个对象（如: `dataset/dataset_generator`），最后将这个实例化对象传入`GeneratorDataset`(mindspore用法)/`DataLoader`(pytorch用法)，至此即可以完成自定义数据集加载了。而MindSpore在`GeneratorDataset`的基础上提供了进一步的`map`->`batch`操作，可以很方便的让用户在`map`内添加一些其他的自定义操作，并将其`batch`起来。
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

A: 关于脚本或者模型迁移，可以查询MindSpore官网中关于[迁移指南](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/overview.html)的介绍。
