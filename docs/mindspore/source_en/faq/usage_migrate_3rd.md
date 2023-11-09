# Migration from a Third-party Framework

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/faq/usage_migrate_3rd.md)

<font size=3>**Q: How do I load a pre-trained PyTorch model for fine-tuning on MindSpore?**</font>

A: Map parameters of PyTorch and MindSpore one by one. No unified conversion script is provided due to flexible network definitions.

In general, the parameters names and parameters values are saved in the CheckPoint file. After invoking the loading interface of the corresponding framework and obtaining the parameter names and values, construct the object according to the MindSpore format, and then you can directly invoke the MindSpore interface to save as CheckPoint files in the MindSpore format.

The main work is to compare the parameter names between different frameworks, so that all parameter names in the network of the two frameworks correspond to each other (a map can be used for mapping). The logic of the following code is transforming the parameter format, excluding the corresponding parameter name.

```python
import torch
import mindspore as ms

def pytorch2mindspore(default_file = 'torch_resnet.pth'):
    """read pth file"""
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

A: For details about script or model migration, please visit the [Migration Guide](https://www.mindspore.cn/docs/en/r2.3/migration_guide/overview.html) in MindSpore official website.
