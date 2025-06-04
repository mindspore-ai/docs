# Model-Related FAQ

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/faq/model_related.md)

## Q: How to deal with network runtime error “Out of Memory” (`OOM`)?

A: First of all, the above error refers to insufficient memory on the device, which may be caused by a variety of reasons, and it is recommended to carry out the following aspects of the investigation.

1. Use the command `npu-smi info` to verify that the card is exclusive.
2. It is recommended to use the default `yaml` configuration for the corresponding network when running network.
3. Increase the value of `max_device_memory` in the corresponding `yaml` configuration file of the network. Note that some memory needs to be reserved for inter-card communication, which can be tried with incremental increases.
4. Adjust the hybrid parallelism strategy, increase pipeline parallelism (pp) and model parallelism (mp) appropriately, and reduce data parallelism (dp) accordingly, keep `dp * mp * pp = device_num`, and increase the number of NPUs if necessary.
5. Try to reduce batch size or sequence length.
6. Turn on selective recalculation or full recalculation, turn on optimizer parallelism.
7. If the problem still needs further troubleshooting, please feel free to [raise issue](https://gitee.com/mindspore/mindformers/issues) for feedback.

<br/>