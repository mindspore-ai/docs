# Migration from a Third-party Framework

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/faq/source_en/usage_migrate_3rd.md)

<font size=3>**Q：How do I load a pre-trained PyTorch model for fine-tuning on MindSpore?**</font>

A：Map parameters of PyTorch and MindSpore one by one. No unified conversion script is provided due to flexible network definitions.
Customize scripts based on scenarios. For details, see [Advanced Usage of Checkpoint](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/advanced_usage_of_checkpoint.html).

<br/>

<font size=3>**Q：How do I convert a PyTorch `dataset` to a MindSpore `dataset`?**</font>

A：The custom dataset logic of MindSpore is similar to that of PyTorch. You need to define a `dataset` class containing `__init__`, `__getitem__`, and `__len__` to read your dataset, instantiate the class into an object (for example, `dataset/dataset_generator`), and transfer the instantiated object to `GeneratorDataset` (on MindSpore) or `DataLoader` (on PyTorch). Then, you are ready to load the custom dataset. MindSpore provides further `map`->`batch` operations based on `GeneratorDataset`. Users can easily add other custom operations to `map` and start `batch`.
The custom dataset of MindSpore is loaded as follows:

```python
# 1. Perform operations such as data argumentation, shuffle, and sampler.
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
# 2. Customize data argumentation.
dataset = dataset.map(operations=pyFunc, …)
# 3. batch
dataset = dataset.batch(batch_size, drop_remainder=True)
```