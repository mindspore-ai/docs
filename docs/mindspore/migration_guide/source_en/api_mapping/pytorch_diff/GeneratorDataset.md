# Function Differences with torch.utils.data.Dataset

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/GeneratorDataset.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.utils.data.Dataset

```python
class torch.utils.data.Dataset
```

For more information, see  [torch.utils.data.Dataset](https://pytorch.org/docs/1.9.0/data.html?highlight=torch%20utils%20data%20dataset#torch.utils.data.Dataset).

## mindspore.dataset.GeneratorDataset

```python
class mindspore.dataset.GeneratorDataset(
    source,
    column_names=None,
    column_types=None,
    schema=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    python_multiprocessing=True,
    max_rowsize=6
    )
```

For more information, see  [mindspore.dataset.GeneratorDataset](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset).

## Differences

PyTorch: An abstract class representing a Dataset. All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite `__getitem__()`, supporting fetching a data sample for a given key. Subclasses could also optionally overwrite `__len__()`, which is expected to return the size of the dataset by many Sampler implementations and the default options of DataLoader.

MindSpore：A source dataset that generates data from Python by invoking Python data source each epoch. The column names and column types of generated dataset depend on Python data defined by users.

## Code Example

```python
import numpy as np
import mindspore.dataset as ds
from torch.utils.data import Dataset

# In MindSpore, GeneratorDataset generates data from Python by invoking Python data source each epoch. The column names and column types of generated dataset depend on Python data defined by users.

class GetDatasetGenerator:

    def __init__(self):
        np.random.seed(58)
        self.__data = np.random.sample((5, 2))
        self.__label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return (self.__data[index], self.__label[index])

    def __len__(self):
        return len(self.__data)


dataset_generator = GetDatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)

for data in dataset.create_dict_iterator():
    print(data["data"], data["label"])
# Out:
# [0.36510558 0.45120592] [0.78888122]
# [0.49606035 0.07562207] [0.38068183]
# [0.57176158 0.28963401] [0.16271622]
# [0.30880446 0.37487617] [0.54738768]
# [0.81585667 0.96883469] [0.77994068]


# In torch, the subclass of torch.utils.data.Dataset should overwrite `__getitem__()`, supporting fetching a data sample for a given key. Subclasses could also optionally overwrite `__len__()`, which is expected to return the size of the dataset.

class GetDatasetGenerator1(Dataset):

    def __init__(self):
        np.random.seed(58)
        self.__data = np.random.sample((5, 2))
        self.__label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return (self.__data[index], self.__label[index])

    def __len__(self):
        return len(self.__data)

dataset = GetDatasetGenerator1()
for item in dataset:
    print("item:", item)

# Out:
# item: (array([0.36510558, 0.45120592]), array([0.78888122]))
# item: (array([0.49606035, 0.07562207]), array([0.38068183]))
# item: (array([0.57176158, 0.28963401]), array([0.16271622]))
# item: (array([0.30880446, 0.37487617]), array([0.54738768]))
# item: (array([0.81585667, 0.96883469]), array([0.77994068]))
```