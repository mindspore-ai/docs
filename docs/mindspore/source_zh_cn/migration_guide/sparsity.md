# 稀疏

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/sparsity.md)

[稀疏张量](https://matteding.github.io/2019/04/25/sparse-matrices/) 是一种特殊张量，其中绝大部分元素的值为零。

在某些应用场景中（比如推荐系统、分子动力学、图神经网络等），数据的特征是稀疏的，若使用普通张量表征这些数据会引入大量不必要的计算、存储和通讯开销。在这种时候就可以使用稀疏张量来表征这些数据。

MindSpore现在已经支持最常用的[CSR和COO](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/tensor.html#%E7%A8%80%E7%96%8F%E5%BC%A0%E9%87%8F)两种稀疏数据格式。但是由于目前支持稀疏算子有限，大部分稀疏的特性还存在限制，在此情况下，建议优先查找对应的算子是否支持稀疏计算，如不支持的话需要转换成普通算子。
由于转换成稠密算子后使用的显存会增加，可能不能使用参考实现的batch size进行训练，此时可以使用 [梯度累加](https://www.mindspore.cn/tutorials/experts/zh-CN/master/optimize/gradient_accumulation.html) 来模拟大batch训练。
