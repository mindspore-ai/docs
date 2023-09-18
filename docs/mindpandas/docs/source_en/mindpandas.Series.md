# mindpandas.Series

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindpandas/docs/source_en/mindpandas.Series.md)&nbsp;&nbsp;

Series is a one-dimensional data structure with axis labels, and is a commonly-used MindSpore Pandas data structure.

## Series API

| MindSpore Pandas Series API       | Pandas API                                                                                                                                            | Supported Platform |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| mindpandas.Series.add       | [pandas.Series.add](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.add.html#pandas.Series.add)                       | CPU                 |
| mindpandas.Series.all       | [pandas.Series.all](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.all.html#pandas.Series.all)                       | CPU                 |
| mindpandas.Series.any       | [pandas.Series.any](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.any.html#pandas.Series.any)                       | CPU                 |
| mindpandas.Series.apply     | [pandas.Series.apply](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.apply.html#pandas.Series.apply)                 | CPU                 |
| mindpandas.Series.astype    | [pandas.Series.astype](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.astype.html#pandas.Series.astype)              | CPU                 |
| mindpandas.Series.copy      | [pandas.Series.copy](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.copy.html#pandas.Series.copy)                    | CPU                 |
| mindpandas.Series.count     | [pandas.Series.count](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.count.html#pandas.Series.count)                 | CPU                 |
| mindpandas.Series.cummax    | [pandas.Series.cummax](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.cummax.html#pandas.Series.cummax)              | CPU                 |
| mindpandas.Series.cummin    | [pandas.Series.cummin](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.cummin.html#pandas.Series.cummin)              | CPU                 |
| mindpandas.Series.cumsum    | [pandas.Series.cumsum](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.cumsum.html#pandas.Series.cumsum)              | CPU                 |
| mindpandas.Series.div       | [pandas.Series.div](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.div.html#pandas.Series.div)                       | CPU                 |
| mindpandas.Series.dtypes    | [pandas.Series.dtypes](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.dtypes.html#pandas.Series.dtypes)              | CPU                 |
| mindpandas.Series.empty     | [pandas.Series.empty](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.empty.html#pandas.Series.empty)                 | CPU                 |
| mindpandas.Series.eq        | [pandas.Series.eq](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.eq.html#pandas.Series.eq)                          | CPU                 |
| mindpandas.Series.equals    | [pandas.Series.equals](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.equals.html#pandas.Series.equals)              | CPU                 |
| mindpandas.Series.fillna    | [pandas.Series.fillna](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.fillna.html#pandas.Series.fillna)              | CPU                 |
| mindpandas.Series.ge        | [pandas.Series.ge](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.ge.html#pandas.Series.ge)                          | CPU                 |
| mindpandas.Series.groupby   | [pandas.Series.groupby](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.groupby.html#pandas.Series.groupby)           | CPU                 |
| mindpandas.Series.gt        | [pandas.Series.gt](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.gt.html#pandas.Series.gt)                          | CPU                 |
| mindpandas.Series.index     | [pandas.Series.index](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.index.html#pandas.Series.index)                 | CPU                 |
| mindpandas.Series.isin      | [pandas.Series.isin](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.isin.html#pandas.Series.isin)                    | CPU                 |
| mindpandas.Series.item      | [pandas.Series.item](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.item.html#pandas.Series.item)                    | CPU                 |
| mindpandas.Series.le        | [pandas.Series.le](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.le.html#pandas.Series.le)                          | CPU                 |
| mindpandas.Series.lt        | [pandas.Series.lt](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.lt.html#pandas.Series.lt)                          | CPU                 |
| mindpandas.Series.max       | [pandas.Series.max](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.max.html#pandas.Series.max)                       | CPU                 |
| mindpandas.Series.mean      | [pandas.Series.mean](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.mean.html#pandas.Series.mean)                    | CPU                 |
| mindpandas.Series.min       | [pandas.Series.min](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.min.html#pandas.Series.min)                       | CPU                 |
| mindpandas.Series.mul       | [pandas.Series.mul](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.mul.html#pandas.Series.mul)                       | CPU                 |
| mindpandas.Series.ne        | [pandas.Series.ne](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.ne.html#pandas.Series.ne)                          | CPU                 |
| mindpandas.Series.prod      | [pandas.Series.prod](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.prod.html#pandas.Series.prod)                    | CPU                 |
| mindpandas.Series.shape     | [pandas.Series.shape](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.shape.html#pandas.Series.shape)                 | CPU                 |
| mindpandas.Series.size      | [pandas.Series.size](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.size.html#pandas.Series.size)                    | CPU                 |
| mindpandas.Series.squeeze   | [pandas.Series.squeeze](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.squeeze.html#pandas.Series.squeeze)           | CPU                 |
| mindpandas.Series.std       | [pandas.Series.std](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.std.html#pandas.Series.std)                       | CPU                 |
| mindpandas.Series.sub       | [pandas.Series.sub](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.sub.html#pandas.Series.sub)                       | CPU                 |
| mindpandas.Series.sum       | [pandas.Series.sum](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.sum.html#pandas.Series.sum)                       | CPU                 |
| mindpandas.Series.tolist    | [pandas.Series.tolist](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.tolist.html#pandas.Series.tolist)              | CPU                 |
| mindpandas.Series.to_dict   | [pandas.Series.to_dict](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.to_dict.html#pandas.Series.to_dict)           | CPU                 |
| mindpandas.Series.to_frame  | [pandas.Series.to_frame](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.to_frame.html#pandas.Series.to_frame)        | CPU                 |
| mindpandas.Series.to_numpy  | [pandas.Series.to_numpy](https://pandas.pydata.org/pandas-docs/version/1.3.5/reference/api/pandas.Series.to_numpy.html#pandas.Series.to_numpy)        | CPU                 |
| mindpandas.Series.to_pandas |                                                                                                                                                       | CPU                 |
