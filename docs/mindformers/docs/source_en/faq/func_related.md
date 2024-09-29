# Function_releated

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/faq/func_related.md)

## Q: How Do I Generate a Model Sharding Strategy File?

A: The model sharding strategy file documents the sharding strategy for model weights in distributed scenarios and is generally used when slicing weights offline. Configure `only_save_strategy: True` in the network `yaml` file, and then start the distributed task normally, then the distributed strategy file can be generated in the `output/strategy/` directory. For details, please refer to the [Tutorial on Slicing and Merging Distributed Weights](https://www.mindspore.cn/mindformers/docs/en/dev/function/transform_weight.html).

<br/>

## Q: How Can I Do When `socket.gaierror: [Errno -2] Name or service not known` or `socket.gaierror: [Errno -3] Temporary failure in name resolution` is Reported in `ranktable` Generation File?

A: Starting from `MindFormers r1.2.0` version, cluster startup is unified using `msrun` method, and `ranktable` startup method is deprecated.

<br/>