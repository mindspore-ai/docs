# Platform and System

`Linux` `Windows` `Ascend` `GPU` `CPU` `Hardware Support` `Beginner` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_en/platform_and_system.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q: Can MindSpore be installed on Ascend 310?

A: Ascend 310 can only be used for inference. MindSpore supports training on Ascend 910. The trained model can be converted into an .om model for inference on Ascend 310.

<br/>

Q: Does MindSpore require computing units such as GPUs and NPUs? What hardware support is required?

A: MindSpore currently supports CPU, GPU, Ascend, and NPU. Currently, you can try out MindSpore through Docker images on laptops or in environments with GPUs. Some models in MindSpore Model Zoo support GPU-based training and inference, and other models are being improved. For distributed parallel training, MindSpore supports multi-GPU training. You can obtain the latest information from [Road Map](https://www.mindspore.cn/doc/note/en/master/roadmap.html) and [project release notes](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md#).

<br/>

Q: Does MindSpore have any plan on supporting other types of heterogeneous computing hardwares?

A: MindSpore provides pluggable device management interface so that developer could easily integrate other types of heterogeneous computing hardwares like FPGA to MindSpore. We welcome more backend support in MindSpore from the community.

<br/>

Q: Does MindSpore support Windows 10?

A: The MindSpore CPU version can be installed on Windows 10. For details about the installation procedure, please refer to the [MindSpore official website tutorial](https://www.mindspore.cn/install/en)

<br/>

Q: For Ascend users, what should I do when `RuntimeError: json.exception.parse_error.101 parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'T'` appears in personal Conda environment?

A: When you encounter the error, you should update the `te/topi/hccl` python toolkits, unload them firstly and then using command `pip install /usr/local/Ascend/fwkacllib/lib64/{te/topi/hccl}*any.whl` to reinstall.