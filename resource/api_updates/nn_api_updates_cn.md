# mindspore.nn API接口变更

与上一版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.default_generator](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/nn/mindspore.nn.default_generator.html#mindspore.nn.default_generator)|New|默认的生成器对象。|r2.3.0rc2: Ascend/GPU/CPU|Default Generator函数
[mindspore.nn.get_rng_state](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/nn/mindspore.nn.get_rng_state.html#mindspore.nn.get_rng_state)|New|获取默认生成器状态。|r2.3.0rc2: Ascend/GPU/CPU|Default Generator函数
[mindspore.nn.initial_seed](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/nn/mindspore.nn.initial_seed.html#mindspore.nn.initial_seed)|New|返回默认生成器的初始种子。|r2.3.0rc2: Ascend/GPU/CPU|Default Generator函数
[mindspore.nn.manual_seed](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/nn/mindspore.nn.manual_seed.html#mindspore.nn.manual_seed)|New|设置默认生成器种子。|r2.3.0rc2: Ascend/GPU/CPU|Default Generator函数
[mindspore.nn.seed](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/nn/mindspore.nn.seed.html#mindspore.nn.seed)|New|生成可作为默认生成器种子的随机种子。|r2.3.0rc2: Ascend/GPU/CPU|Default Generator函数
[mindspore.nn.set_rng_state](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/nn/mindspore.nn.set_rng_state.html#mindspore.nn.set_rng_state)|New|设置默认生成器状态。|r2.3.0rc2: Ascend/GPU/CPU|Default Generator函数
[mindspore.nn.Generator](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/nn/mindspore.nn.Generator.html#mindspore.nn.Generator)|New|管理随机数状态的生成器，为随机函数提供seed和offset。|r2.3.0rc2: Ascend/GPU/CPU|Generator类
