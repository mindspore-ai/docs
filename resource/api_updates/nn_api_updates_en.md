# mindspore.nn API Interface Change

Compared with the version 2.9.0, the added, deleted and supported platforms change information of `mindspore.nn` operators in version 2.10.0, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.nn.FTRL](https://mindspore.cn/docs/en/r2.10.0/api_python/nn/mindspore.nn.FTRL.html#mindspore.nn.FTRL)|Changed|r2.9.0: Implements the FTRL algorithm. => r2.10.0: mindspore.nn.FTRL is deprecated from version 2.9.0 and will be removed in a future version.|r2.9.0: Ascend/GPU => r2.10.0: |Optimizer
[mindspore.nn.LARS](https://mindspore.cn/docs/en/r2.10.0/api_python/nn/mindspore.nn.LARS.html#mindspore.nn.LARS)|Changed|r2.9.0: Implements the LARS algorithm. => r2.10.0: mindspore.nn.LARS is deprecated from version 2.9.0 and will be removed in a future version.|r2.9.0: Ascend => r2.10.0: |Optimizer
[mindspore.nn.ProximalAdagrad](https://mindspore.cn/docs/en/r2.10.0/api_python/nn/mindspore.nn.ProximalAdagrad.html#mindspore.nn.ProximalAdagrad)|Changed|r2.9.0: Implements the ProximalAdagrad algorithm that is an online Learning and Stochastic Optimization. => r2.10.0: mindspore.nn.ProximalAdagrad is deprecated from version 2.9.0 and will be removed in a future version.|r2.9.0: Ascend/GPU => r2.10.0: |Optimizer
