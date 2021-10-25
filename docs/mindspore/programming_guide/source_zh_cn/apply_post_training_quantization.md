# 应用训练后量化

`端侧` `Ascend` `推理应用`

<!-- TOC -->

- [应用训练后量化](#应用训练后量化)
    - [概念](#概念)
        - [权重量化](#权重量化)
        - [全量化](#全量化)
    - [训练后量化工具](#训练后量化工具)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/apply_post_training_quantization.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概念

训练后量化是指对预训练后的网络进行权重量化或者全量化，以达到减小模型大小和提升推理性能的目的。
该过程不需要训练，对激活值量化时需要少量校准数据。

### 权重量化

对模型的权值进行量化，仅压缩模型大小，推理时仍然执行float32运算。量化比特数越低，模型压缩率越大，但是精度损失通常也比较大。

### 全量化

对模型的权重和激活值统一进行量化，推理时执行int运算。可以减小模型大小、提升模型推理速度和降低功耗。
针对需要提升模型运行速度、降低模型运行功耗的场景，可以使用训练后全量化功能。为了计算激活值的量化参数，用户需要提供校准数据集。

## 训练后量化工具

根据模型推理部署的硬件平台选择使用对应的训练后量化工具。

| 训练后量化工具 | 量化方法支持 | 推理硬件平台支持 | 量化模型部署 |
| --- | --- | --- | --- |
| [MindSpore训练后量化工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html) | 权重量化 <br> 全量化 | CPU | [端侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/use/runtime.html) |
| 昇腾模型压缩工具 | 全量化 | Ascend 310 AI处理器 | [Ascend 310 AI处理器上推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_ascend_310.html) |
