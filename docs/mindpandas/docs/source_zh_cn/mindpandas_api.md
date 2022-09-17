# mindpandas API参考

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindpandas/docs/source_zh_cn/mindpandas_api.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

MindPandas 工具包提供了一系列类Pandas接口。用户可以使用类Pandas语法在数据集上进行数据处理。

MindPandas API可以分为四大类：mindpandas.DataFrame类API，mindpandas.Series类API，mindpandas.groupby类API，以及其他类API。

在API示例中，模块导入方法如下:

```python
import mindpandas as pd
```

## MindPandas全局参数设置操作

为了调整MindPandas的缺省设置，我们可以通过命令设置config参数。

```python
from mindpandas import config
config.set_concurrency_mode(mode='multithread') # MindPandas will use multithread mode

config.set_partition_shape(shape=(2,2)) # MindPandas will set the partition shape with (2,2)
```

### set_concurrency_mode(mode)

设置后端运行模式。

参数含义如下：

mode (str)：可以设置为multithread或yr两种后端模式，默认值为multithread。multithread模式为多线程后端，yr模式为多进程后端。

异常情况：

ValueError: 该模式不支持。

### set_partition_shape(shape)

设置并行计算时的切片维度。

参数含义如下：

shape (tuple)：切片维度，数据类型为2的倍数组成的tuple。默认值为16 * 16。

异常情况：

ValueError: 每个切片的维度只能是元组类型。

## to_pandas接口

当前MindPandas还有部分接口暂未支持。但是我们可以通过`to_pandas`方法来适配MindPandas和原生Pandas之间数据类型的转换。
使用时，在API的执行完成后，加上to_pandas函数接口，便可以把MindPandas的DataFrame格式转换成Pandas的DataFrame格式，MindPandas的Series格式转换成Pandas的Series格式，后续可以使用其他Pandas的API接口进行数据处理。
MindPandas的Groupby接口不支持使用to_pandas。

```python
import mindpandas as pd

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df_pandas = df.to_pandas()
```

## DataFrame 类操作

对于二维或者多维度的数据，MindPandas会读取到DataFrame类并进行处理，通过字典构建DataFrame示例：

```python
import mindpandas as pd
import numpy as np
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
print(df)
```

运行结果如下：

```python
   col1  col2
0     1     3
1     2     4
```

生成指定dtype：

```python
df = pd.DataFrame(data=d, dtype=np.int8)
print(df.dtypes)
```

运行结果如下：

```python
col1    int8
col2    int8
dtype: object
```

从numpy ndarray构造DataFrame：

```python
df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])
print(df2)
```

运行结果如下：

```python
   a  b  c
0  1  2  3
1  4  5  6
2  7  8  9
```

| MindPandas DataFrame API | Pandas API                                                                                                                                                     | 支持平台 | 说明 |
|--------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|----------------------------------|
| mindpandas.DataFrame.add                | [pandas.DataFrame.add](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.add.html#pandas.DataFrame.add)                                     | CPU                 |                                  |
| mindpandas.DataFrame.all                | [pandas.DataFrame.all](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.all.html#pandas.DataFrame.all)                                     | CPU                 |                                  |
| mindpandas.DataFrame.any                | [pandas.DataFrame.any](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.any.html#pandas.DataFrame.any)                                     | CPU                 |                                  |
| mindpandas.DataFrame.apply              | [pandas.DataFrame.apply](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply)                               | CPU                 |                                  |
| mindpandas.DataFrame.applymap           | [pandas.DataFrame.applymap](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.applymap.html#pandas.DataFrame.applymap)                      | CPU                 |                                  |
| mindpandas.DataFrame.astype             | [pandas.DataFrame.astype](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.astype.html#pandas.DataFrame.astype)                            | CPU                 |                                  |
| mindpandas.DataFrame.columns            | [pandas.DataFrame.columns](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.columns.html#pandas.DataFrame.columns)                         | CPU                 |                                  |
| mindpandas.DataFrame.combine            | [pandas.DataFrame.combine](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.combine.html#pandas.DataFrame.combine)                         | CPU                 |                                  |
| mindpandas.DataFrame.copy               | [pandas.DataFrame.copy](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.copy.html#pandas.DataFrame.copy)                                  | CPU                 |                                  |
| mindpandas.DataFrame.cumsum             | [pandas.DataFrame.cumsum](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.cumsum.html#pandas.DataFrame.cumsum)                            | CPU                 |                                  |
| mindpandas.DataFrame.div                | [pandas.DataFrame.div](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.div.html#pandas.DataFrame.div)                                     | CPU                 |                                  |
| mindpandas.DataFrame.drop               | [pandas.DataFrame.drop](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.drop.html#pandas.DataFrame.drop)                                  | CPU                 |                                  |
| mindpandas.DataFrame.drop_duplicates    | [pandas.DataFrame.drop_duplicates](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.drop_duplicates.html#pandas.DataFrame.drop_duplicates) | CPU                 |                                  |
| mindpandas.DataFrame.dropna             | [pandas.DataFrame.dropna](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna)                            | CPU                 |                                  |
| mindpandas.DataFrame.dtypes             | [pandas.DataFrame.dtypes](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.dtypes.html#pandas.DataFrame.dtypes)                            | CPU                 |                                  |
| mindpandas.DataFrame.duplicated         | [pandas.DataFrame.duplicated](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.duplicated.html#pandas.DataFrame.duplicated)                | CPU                 |                                  |
| mindpandas.DataFrame.empty              | [pandas.DataFrame.empty](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.empty.html#pandas.DataFrame.empty)                               | CPU                 |                                  |
| mindpandas.DataFrame.eq                 | [pandas.DataFrame.eq](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.eq.html#pandas.DataFrame.eq)                                        | CPU                 |                                  |
| mindpandas.DataFrame.fillna             | [pandas.DataFrame.fillna](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna)                            | CPU                 |                                  |
| mindpandas.DataFrame.ge                 | [pandas.DataFrame.ge](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.ge.html#pandas.DataFrame.ge)                                        | CPU                 |                                  |
| mindpandas.DataFrame.groupby            | [pandas.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby)                         | CPU                 |                                  |
| mindpandas.DataFrame.gt                 | [pandas.DataFrame.gt](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.gt.html#pandas.DataFrame.gt)                                        | CPU                 |                                  |
| mindpandas.DataFrame.head               | [pandas.DataFrame.head](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.head.html#pandas.DataFrame.head)                                  | CPU                 |                                  |
| mindpandas.DataFrame.iloc               | [pandas.DataFrame.iloc](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc)                                  | CPU                 |                                  |
| mindpandas.DataFrame.index              | [pandas.DataFrame.index](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.index.html#pandas.DataFrame.index)                               | CPU                 |                                  |
| mindpandas.DataFrame.insert             | [pandas.DataFrame.insert](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.insert.html#pandas.DataFrame.insert)                            | CPU                 |                                  |
| mindpandas.DataFrame.isin               | [pandas.DataFrame.isin](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.isin.html#pandas.DataFrame.isin)                                  | CPU                 |                                  |
| mindpandas.DataFrame.isna               | [pandas.DataFrame.isna](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.isna.html#pandas.DataFrame.isna)                                  | CPU                 |                                  |
| mindpandas.DataFrame.iterrows           | [pandas.DataFrame.iterrows](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows)                      | CPU                 |                                  |
| mindpandas.DataFrame.le                 | [pandas.DataFrame.le](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.le.html#pandas.DataFrame.le)                                        | CPU                 |                                  |
| mindpandas.DataFrame.loc                | [pandas.DataFrame.loc](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.loc.html#pandas.DataFrame.loc)                                     | CPU                 |                                  |
| mindpandas.DataFrame.lt                 | [pandas.DataFrame.lt](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.lt.html#pandas.DataFrame.lt)                                        | CPU                 |                                  |
| mindpandas.DataFrame.max                | [pandas.DataFrame.max](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.max.html#pandas.DataFrame.max)                                     | CPU                 |                                  |
| mindpandas.DataFrame.mean               | [pandas.DataFrame.mean](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.mean.html#pandas.DataFrame.mean)                                  | CPU                 |                                  |
| mindpandas.DataFrame.median             | [pandas.DataFrame.median](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.median.html#pandas.DataFrame.median)                            | CPU                 |                                  |
| mindpandas.DataFrame.merge              | [pandas.DataFrame.merge](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge)                               | CPU                 |                                  |
| mindpandas.DataFrame.min                | [pandas.DataFrame.min](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.min.html#pandas.DataFrame.min)                                     | CPU                 |                                  |
| mindpandas.DataFrame.mul                | [pandas.DataFrame.mul](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.mul.html#pandas.DataFrame.mul)                                     | CPU                 |                                  |
| mindpandas.DataFrame.ne                 | [pandas.DataFrame.ne](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.ne.html#pandas.DataFrame.ne)                                        | CPU                 |                                  |
| mindpandas.DataFrame.rename             | [pandas.DataFrame.rename](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.rename.html#pandas.DataFrame.rename)                            | CPU                 |                                  |
| mindpandas.DataFrame.reset_index        | [pandas.DataFrame.reset_index](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.reset_index.html#pandas.DataFrame.reset_index)             | CPU                 |                                  |
| mindpandas.DataFrame.shape              | [pandas.DataFrame.shape](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.shape.html#pandas.DataFrame.shape)                               | CPU                 |                                  |
| mindpandas.DataFrame.sort_values        | [pandas.DataFrame.sort_values](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.sort_values.html#pandas.DataFrame.sort_values)             | CPU                 |                                  |
| mindpandas.DataFrame.squeeze            | [pandas.DataFrame.squeeze](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.squeeze.html#pandas.DataFrame.squeeze)                         | CPU                 |                                  |
| mindpandas.DataFrame.std                | [pandas.DataFrame.std](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.std.html#pandas.DataFrame.std)                                     | CPU                 |                                  |
| mindpandas.DataFrame.sub                | [pandas.DataFrame.sub](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.sub.html#pandas.DataFrame.sub)                                     | CPU                 |                                  |
| mindpandas.DataFrame.sum                | [pandas.DataFrame.sum](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.sum.html#pandas.DataFrame.sum)                                     | CPU                 |                                  |
| mindpandas.DataFrame.tail               | [pandas.DataFrame.tail](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.tail.html#pandas.DataFrame.tail)                                  | CPU                 |                                  |
| mindpandas.DataFrame.transpose          | [pandas.DataFrame.transpose](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.transpose.html#pandas.DataFrame.transpose)                   | CPU                 |                                  |
| mindpandas.DataFrame.values             | [pandas.DataFrame.values](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.DataFrame.values.html#pandas.DataFrame.values)                            | CPU                 |                                  |

## Series 类操作

对于一维的数据，MindPandas会读取到Series类并进行处理，Series类支持的API如下：

| MindPandas Series API | Pandas API                                                                                                                         | 支持平台 | 说明 |
| --------------- |-----------------------------------------------------------------------------------------------------------------------------------------|---------------------| -------------------------------- |
| mindpandas.Series.add             | [pandas.Series.add](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.add.html#pandas.Series.add)                | CPU                 |                                  |
| mindpandas.Series.all             | [pandas.Series.all](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.all.html#pandas.Series.all)                | CPU                 |                                  |
| mindpandas.Series.any             | [pandas.Series.any](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.any.html#pandas.Series.any)                | CPU                 |                                  |
| mindpandas.Series.apply           | [pandas.Series.apply](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.apply.html#pandas.Series.apply)          | CPU                 |                                  |
| mindpandas.Series.astype          | [pandas.Series.astype](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.astype.html#pandas.Series.astype)       | CPU                 |                                  |
| mindpandas.Series.copy            | [pandas.Series.copy](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.copy.html#pandas.Series.copy)             | CPU                 |                                  |
| mindpandas.Series.cumsum          | [pandas.Series.cumsum](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.cumsum.html#pandas.Series.cumsum)       | CPU                 |                                  |
| mindpandas.Series.div             | [pandas.Series.div](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.div.html#pandas.Series.div)                | CPU                 |                                  |
| mindpandas.Series.dtypes          | [pandas.Series.dtypes](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.dtypes.html#pandas.Series.dtypes)       | CPU                 |                                  |
| mindpandas.Series.empty           | [pandas.Series.empty](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.empty.html#pandas.Series.empty)          | CPU                 |                                  |
| mindpandas.Series.eq              | [pandas.Series.eq](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.eq.html#pandas.Series.eq)                   | CPU                 |                                  |
| mindpandas.Series.fillna          | [pandas.Series.fillna](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.fillna.html#pandas.Series.fillna)       | CPU                 |                                  |
| mindpandas.Series.ge              | [pandas.Series.ge](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.ge.html#pandas.Series.ge)                   | CPU                 |                                  |
| mindpandas.Series.groupby         | [pandas.Series.groupby](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.groupby.html#pandas.Series.groupby)    | CPU                 |                                  |
| mindpandas.Series.gt              | [pandas.Series.gt](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.gt.html#pandas.Series.gt)                   | CPU                 |                                  |
| mindpandas.Series.index           | [pandas.Series.index](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.index.html#pandas.Series.index)          | CPU                 |                                  |
| mindpandas.Series.le              | [pandas.Series.le](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.le.html#pandas.Series.le)                   | CPU                 |                                  |
| mindpandas.Series.lt              | [pandas.Series.lt](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.lt.html#pandas.Series.lt)                   | CPU                 |                                  |
| mindpandas.Series.max             | [pandas.Series.max](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.max.html#pandas.Series.max)                | CPU                 |                                  |
| mindpandas.Series.mean            | [pandas.Series.mean](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.mean.html#pandas.Series.mean)             | CPU                 |                                  |
| mindpandas.Series.min             | [pandas.Series.min](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.min.html#pandas.Series.min)                | CPU                 |                                  |
| mindpandas.Series.mul             | [pandas.Series.mul](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.mul.html#pandas.Series.mul)                | CPU                 |                                  |
| mindpandas.Series.ne              | [pandas.Series.ne](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.ne.html#pandas.Series.ne)                   | CPU                 |                                  |
| mindpandas.Series.shape           | [pandas.Series.shape](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.shape.html#pandas.Series.shape)          | CPU                 |                                  |
| mindpandas.Series.size            | [pandas.Series.size](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.size.html#pandas.Series.size)             | CPU                 |                                  |
| mindpandas.Series.squeeze         | [pandas.Series.squeeze](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.squeeze.html#pandas.Series.squeeze)    | CPU                 |                                  |
| mindpandas.Series.std             | [pandas.Series.std](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.std.html#pandas.Series.std)                | CPU                 |                                  |
| mindpandas.Series.sub             | [pandas.Series.sub](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.sub.html#pandas.Series.sub)                | CPU                 |                                  |
| mindpandas.Series.sum             | [pandas.Series.sum](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.sum.html#pandas.Series.sum)                | CPU                 |                                  |
| mindpandas.Series.to_dict         | [pandas.Series.to_dict](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.to_dict.html#pandas.Series.to_dict)    | CPU                 |                                  |
| mindpandas.Series.to_frame        | [pandas.Series.to_frame](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.to_frame.html#pandas.Series.to_frame) | CPU                 |                                  |
| mindpandas.Series.to_numpy        | [pandas.Series.to_numpy](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.to_numpy.html#pandas.Series.to_numpy) | CPU                 |                                  |
| mindpandas.Series.to_list         | [pandas.Series.to_list](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.to_list.html#pandas.Series.to_list)    | CPU                 |                                  |

## Groupby 类操作

| MindPandas Groupby API | Pandas API                                                                                                                                                                | 支持平台 | 说明 |
| ---------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------------------- | -------------------------------- |
| mindpandas.GroupBy.all              | [pandas.core.groupby.GroupBy.all](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.all.html#pandas.core.groupby.GroupBy.all)             | CPU                 |                                  |
| mindpandas.GroupBy.any              | [pandas.core.groupby.GroupBy.any](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.any.html#pandas.core.groupby.GroupBy.any)             | CPU                 |                                  |
| mindpandas.GroupBy.count            | [pandas.core.groupby.GroupBy.count](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.count.html#pandas.core.groupby.GroupBy.count)       | CPU                 |                                  |
| mindpandas.GroupBy.groups           | [pandas.core.groupby.GroupBy.groups](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.groups.html#pandas.core.groupby.GroupBy.groups)    | CPU                 |                                  |
| mindpandas.GroupBy.indices          | [pandas.core.groupby.GroupBy.indices](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.indices.html#pandas.core.groupby.GroupBy.indices) | CPU                 |                                  |
| mindpandas.GroupBy.max              | [pandas.core.groupby.GroupBy.max](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.max.html#pandas.core.groupby.GroupBy.max)             | CPU                 |                                  |
| mindpandas.GroupBy.min              | [pandas.core.groupby.GroupBy.min](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.min.html#pandas.core.groupby.GroupBy.min)             | CPU                 |                                  |
| mindpandas.GroupBy.ngroup           | [pandas.core.groupby.GroupBy.ngroup](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.ngroup.html#pandas.core.groupby.GroupBy.ngroup)    | CPU                 |                                  |
| mindpandas.GroupBy.prod             | [pandas.core.groupby.GroupBy.prod](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.prod.html#pandas.core.groupby.GroupBy.prod)          | CPU                 |                                  |
| mindpandas.GroupBy.size             | [pandas.core.groupby.GroupBy.size](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.size.html#pandas.core.groupby.GroupBy.size)          | CPU                 |                                  |
| mindpandas.GroupBy.sum              | [pandas.core.groupby.GroupBy.sum](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.core.groupby.GroupBy.sum.html#pandas.core.groupby.GroupBy.sum)             | CPU                 |                                  |

## Other 类操作

| MindPandas Other API | Pandas API                                                                                                                              | 支持平台 | 说明 |
| -------------- |-----------------------------------------------------------------------------------------------------------------------------------------------| ------------------- | -------------------------------- |
| mindpandas.concat         | [pandas.concat](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.concat.html?highlight=concat#pandas.concat)                 | CPU                 |                                  |
| mindpandas.date_range     | [pandas.date_range](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.date_range.html?highlight=date_range#pandas.date_range) | CPU                 |                                  |
| mindpandas.read_csv       | [pandas.read_csv](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.read_csv.html?highlight=read_csv#pandas.read_csv)         | CPU                 |                                  |