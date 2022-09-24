# MindPandas.DataFrame

DataFrame是一个具有行、列索引的二维表数据结构，是MindPandas主要的数据结构之一。

## DataFrame构造

通过字典构建DataFrame示例：

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

指定数据类型：

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

通过numpy ndarray构造DataFrame：

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

## DataFrame API

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
