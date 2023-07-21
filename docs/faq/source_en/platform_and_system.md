# Platform and System

`Linux` `Windows` `Ascend` `GPU` `CPU` `Hardware Support` `Beginner` `Intermediate`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/tree/r1.0/docs/faq/source_en/platform_and_system.md)

<font size=3>**Q: Does MindSpore run only on Huawei `NPUs`?**</font>

A: MindSpore supports Huawei Ascend `NPUs`, `GPUs`, and `CPUs`, and supports heterogeneous computing.

<br/>

<font size=3>**Q: Can MindSpore be converted to an AIR model on Ascend 310?**</font>

A: An AIR model cannot be exported from the Ascend 310. You need to load a trained checkpoint on the Ascend 910, export an AIR model, and then convert the AIR model into an OM model for inference on the Ascend 310. For details about the Ascend 910 installation, see the MindSpore Installation Guide at [here](https://www.mindspore.cn/install/en).

<br/>

<font size=3>**Q: Can a network script trained by MindSpore on a GPU be directly trained on an NPU without modification?**</font>

A: Yes. MindSpore provides unified APIs for NPUs, GPUs, and CPUs. With the support of operators, network scripts can run across platforms without modification.

<br/>

<font size=3>**Q: Can MindSpore be installed on Ascend 310?**</font>

A: Ascend 310 can only be used for inference. MindSpore supports training on Ascend 910. The trained model can be converted into an .om model for inference on Ascend 310.

<br/>

<font size=3>**Q: Does MindSpore require computing units such as GPUs and NPUs? What hardware support is required?**</font>

A: MindSpore currently supports CPU, GPU, Ascend, and NPU. Currently, you can try out MindSpore through Docker images on laptops or in environments with GPUs. Some models in MindSpore Model Zoo support GPU-based training and inference, and other models are being improved. For distributed parallel training, MindSpore supports multi-GPU training. You can obtain the latest information from [Road Map](https://www.mindspore.cn/doc/note/en/r1.0/roadmap.html) and [project release notes](https://gitee.com/mindspore/mindspore/blob/r1.0/RELEASE.md#).

<br/>

<font size=3>**Q: Does MindSpore have any plan on supporting other types of heterogeneous computing hardwares?**</font>

A: MindSpore provides pluggable device management interface so that developer could easily integrate other types of heterogeneous computing hardwares like FPGA to MindSpore. We welcome more backend support in MindSpore from the community.

<br/>

<font size=3>**Q: Does MindSpore support Windows 10?**</font>

A: The MindSpore CPU version can be installed on Windows 10. For details about the installation procedure, please refer to the [MindSpore official website tutorial](https://www.mindspore.cn/install/en)

<br/>

<font size=3>**Q: For Ascend users, what should I do when `RuntimeError: json.exception.parse_error.101 parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'T'` appears in personal Conda environment?**</font>

A: When you encounter the error, you should update the `te/topi/hccl` python toolkits, unload them firstly and then using command `pip install /usr/local/Ascend/fwkacllib/lib64/{te/topi/hccl}*any.whl` to reinstall.