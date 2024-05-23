# mindspore.nn API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.nn.default_generator](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/nn/mindspore.nn.default_generator.html#mindspore.nn.default_generator)|New|Return the default generator object.|r2.3.0rc2: Ascend/GPU/CPU|Default Generator Function
[mindspore.nn.get_rng_state](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/nn/mindspore.nn.get_rng_state.html#mindspore.nn.get_rng_state)|New|Get the default generator state.|r2.3.0rc2: Ascend/GPU/CPU|Default Generator Function
[mindspore.nn.initial_seed](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/nn/mindspore.nn.initial_seed.html#mindspore.nn.initial_seed)|New|Return the initial seed of default generator.|r2.3.0rc2: Ascend/GPU/CPU|Default Generator Function
[mindspore.nn.manual_seed](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/nn/mindspore.nn.manual_seed.html#mindspore.nn.manual_seed)|New|Sets the default generator seed.|r2.3.0rc2: Ascend/GPU/CPU|Default Generator Function
[mindspore.nn.seed](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/nn/mindspore.nn.seed.html#mindspore.nn.seed)|New|Generate random seeds that can be used as seeds for default generator.|r2.3.0rc2: Ascend/GPU/CPU|Default Generator Function
[mindspore.nn.set_rng_state](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/nn/mindspore.nn.set_rng_state.html#mindspore.nn.set_rng_state)|New|Sets the default generator state.|r2.3.0rc2: Ascend/GPU/CPU|Default Generator Function
[mindspore.nn.Generator](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/nn/mindspore.nn.Generator.html#mindspore.nn.Generator)|New|A generator that manages the state of random numbers and provides seed and offset for random functions.|r2.3.0rc2: Ascend/GPU/CPU|Generator Class
