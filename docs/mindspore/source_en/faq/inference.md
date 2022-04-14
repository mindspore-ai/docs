# Inference

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/faq/inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: When MindSpore 1.3 is installed on the Ascend 310 hardware platform, why an error message is displayed when I run the `add_model.py` sample in mindspore_serving?**</font>

A: Ascend 310 supports model export and Serving inference, but does not support direct inference by using the MindSpore frontend Python script. In the `add` sample, the code for direct inference by using the MindSpore frontend Python script is added in the export model. You only need to comment out the code in the Ascend 310 scenario.

```python
def export_net():
    """Export add net of 2x2 + 2x2, and copy output model `tensor_add.mindir` to directory ../add/1"""
    x = np.ones([2, 2]).astype(np.float32)
    y = np.ones([2, 2]).astype(np.float32)
    add = Net()
    # Comment out the MindSpore frontend Python script used for direct inference in the Ascend 310 scenario.
    # output = add(ms.Tensor(x), ms.Tensor(y))
    ms.export(add, ms.Tensor(x), ms.Tensor(y), file_name='tensor_add', file_format='MINDIR')
    dst_dir = '../add/1'
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass

    dst_file = os.path.join(dst_dir, 'tensor_add.mindir')
    copyfile('tensor_add.mindir', dst_file)
    print("copy tensor_add.mindir to " + dst_dir + " success")

    print(x)
    print(y)
    # print(output.asnumpy()).
```

<br/>

<font size=3>**Q: What should I do when an error `/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found` prompts during application compiling?**</font>

A: Find the directory where the missing dynamic library file is located, add the path to the environment variable `LD_LIBRARY_PATH`, and refer to [Inference Using the MindIR Model on Ascend 310 AI Processors#Building Inference Code](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/ascend_310_mindir.html#building-inference-code) for environment variable settings.

<br/>

<font size=3>**Q: What should I do when error `error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory` prompts during application running?**</font>

A: While Ascend 310 AI Processor software packages relied by MindSpore is installed, the `CANN` package should install the full-featured `toolkit` version instead of the `nnrt` version.

<font size=3>**Q: How to set high-precision or high-performance mode when performing inference on Ascend 310 AI Processor?**</font>

A: Set in the inference code through the SetPrecisionMode interface of Ascend310DeviceInfo. Optional: force_fp16, allow_fp32_to_fp16, must_keep_origin_dtype, and allow_mix_precision. The default value is force_fp16, which refers to the high-performance mode. High precision mode can be set to allow_fp32_to_fp16 or must_keep_origin_dtype.
<br/>

<font size=3>**Q: How to configure AIPP files?**</font>

A: AIPP (artistic intelligence pre-processing) AI preprocessing is used to complete image preprocessing on AI core, including changing image size, color gamut conversion (converting image format), subtracting mean / multiplication coefficient (changing image pixels). Real-time inference is performed after data processing. The related configuration introduction is complex. Please refer to [AIPP enable chapter of ATC tool](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0015.html).
<br/>

<font size=3>**Q: How to set the log level in the inferenct process of Ascend 310 AI Processor?**</font>

A: Use ASCEND_GLOBAL_LOG_LEVEL to set log level, 0: Debug level; 1: Info level; 2: Warning level; 3: Error level; 4: Null level, no log output; Other values are illegal. Configuration example: export ASCEND_GLOBAL_LOG_LEVEL=1. If there are errors in the inference process, you can modify the log level to obtain more detailed log information.
