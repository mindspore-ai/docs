# 第三方框架迁移使用类

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_zh_cn/usage_migrate_3rd.md)

<font size=3>**Q：请问想加载PyTorch预训练好的模型用于MindSpore模型finetune有什么方法？**</font>

A：需要把PyTorch和MindSpore的参数进行一一对应，因为网络定义的灵活性，所以没办法提供统一的转化脚本。
需要根据场景书写定制化脚本，可参考[checkpoint高级用法](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.2/advanced_usage_of_checkpoint.html)

<br/>

<font size=3>**Q：怎么将PyTorch的`dataset`转换成MindSpore的`dataset`？**</font>

A：MindSpore和PyTorch的自定义数据集逻辑是比较类似的，需要用户先定义一个自己的`dataset`类，该类负责定义`__init__`，`__getitem__`,`__len__`来读取自己的数据集，然后将该类实例化为一个对象（如：`dataset/dataset_generator`），最后将这个实例化对象传入`GeneratorDataset`(mindspore用法)/`DataLoader`(pytorch用法)，至此即可以完成自定义数据集加载了。而mindspore在`GeneratorDataset`的基础上提供了进一步的`map`->`batch`操作，可以很方便的让用户在`map`内添加一些其他的自定义操作，并将其`batch`起来。
对应的MindSpore的自定义数据集加载如下：

```python
#1 Data enhancement,shuffle,sampler.
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
#2 Custom data enhancement
dataset = dataset.map(operations=pyFunc, {other_params})
#3 batch
dataset = dataset.batch(batch_size, drop_remainder=True)
```
