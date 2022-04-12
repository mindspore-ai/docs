# 推理

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/faq/inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: 在Ascend310硬件平台安装了MindSpore1.3版本，运行mindspore_serving中的`add_model.py`样例出现报错？**</font>

A: Ascend310上支持模型导出、Serving推理，但是不支持MindSpore前端Python脚本直接推理，`add`样例中导出模型多了MindSpore前端Python脚本直接推理的代码，在Ascend310场景注释掉即可。

```python
def export_net():
    """Export add net of 2x2 + 2x2, and copy output model `tensor_add.mindir` to directory ../add/1"""
    x = np.ones([2, 2]).astype(np.float32)
    y = np.ones([2, 2]).astype(np.float32)
    add = Net()
    # MindSpore前端Python脚本直接推理，310注释掉
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
    # print(output.asnumpy())。
```

<br/>

<font size=3>**Q: 编译应用时报错`/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found`怎么办？**</font>

A: 寻找缺少的动态库文件所在目录，添加该路径到环境变量`LD_LIBRARY_PATH`中，环境变量设置参考[Ascend 310 AI处理器上使用MindIR模型进行推理#编译推理代码](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference_ascend_310_mindir.html#编译推理代码)。

<br/>

<font size=3>**Q: 运行应用时报错`error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory`怎么办？**</font>

A: 安装MindSpore所依赖的Ascend 310 AI处理器配套软件包时，`CANN`包不能安装`nnrt`版本，而是需要安装功能完整的`toolkit`版本。

<br/>

<font size=3>**Q: 使用昇腾310进行推理的时候怎么设置高精度或者高性能模式？**</font>

A: 在推理代码中通过Ascend310DeviceInfo中的SetPrecisionMode接口进行设置，可选：force_fp16、allow_fp32_to_fp16、must_keep_origin_dtype，allow_mix_precision。默认值为force_fp16，指的就是高性能模式。高精度模式可设置为allow_fp32_to_fp16或must_keep_origin_dtype。

<br/>

<font size=3>**Q: AIPP文件怎么配置？**</font>

A: AIPP（Artificial Intelligence Pre-Processing）AI预处理，用于在AI Core上完成图像预处理，包括改变图像尺寸、色域转换（转换图像格式）、减均值/乘系数（改变图像像素），数据处理之后再进行真正的模型推理。相关的配置介绍比较复杂，可以参考[ATC工具的AIPP使能章节](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0015.html)。

<br/>

<font size=3>**Q: 怎么设置昇腾310推理过程中的日志级别？**</font>

A: 通过ASCEND_GLOBAL_LOG_LEVEL来设置，0：DEBUG级别；1：INFO级别；2：WARNING级别；3：ERROR级别；4：NULL级别，不输出日志；其他值为非法值。配置示例：export ASCEND_GLOBAL_LOG_LEVEL=1。如果推理过程中出现错误可通过修改日志级别来获取更详细的日志信息。
