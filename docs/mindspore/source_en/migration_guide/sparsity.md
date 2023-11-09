# Sparsity

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/migration_guide/sparsity.md)

A [sparse tensor](https://matteding.github.io/2019/04/25/sparse-matrices/) is a special tensor in which the value of the most significant element is zero.

In some scenarios (such as recommendation systems, molecular dynamics, graph neural networks), the data is sparse. If you use common dense tensors to represent the data, you may introduce many unnecessary calculations, storage, and communication costs. In this case, it is better to use sparse tensor to represent the data.

MindSpore now supports the most commonly used [CSR and COO data formats](https://www.mindspore.cn/tutorials/en/r2.3/beginner/tensor.html#sparse-tensor). Currently, only a limited number of sparse operators are supported, and most sparse features are restricted. In this case, you are advised to check whether the corresponding operator supports sparse computing. If the operator does not support sparse computing, convert it into a common operator.
After the operator is converted into a dense operator, the video memory used increases. Therefore, the batch size implemented by referring to may not be used for training. In this case, you can use [Gradient Accumulation](https://www.mindspore.cn/tutorials/experts/en/r2.3/optimize/gradient_accumulation.html) to simulate large batch training.
