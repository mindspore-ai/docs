# Applying Post Training Quantization

Translator: [unseeme](https://gitee.com/unseenme)

`Device` `Ascend` `Inference Application`

<!-- TOC -->

- [Applying Post Training Quantization](#applying-post-training-quantization)
    - [Concept](#concept)
        - [Weight Quantization](#weight-quantization)
        - [Full Quantization](#full-quantization)
    - [Post Training Quantization Tools](#post-training-quantization-tools)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_en/apply_post_training_quantization.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Concept

Post training quantization refers to perform weights quantization or full quantization on a pre-trained model. It can reduce model size while also speed up the inference.
This process does not require training. Small amounts of calibration data is needed for activations quantization.

### Weights Quantization

Quantify the weights of the model, only reduce the model size. Float32 operations are still performed during inference. The lower the number of quantization bits, the greater the model compression rate, but accuracy loss is usually become relatively large.

### Full Quantization

Quantify the weights and activations of the model, int operations are performed during inference. It can reduce the size of the model, increase the speed of model inference, and reduce power consumption.
For scenarios that need to increase the running speed and reduce the power consumption of the model, you can use the post training full quantization. In order to calculate the quantitative parameters of the activations, the user needs to provide a calibration dataset.

## Post Training Quantization Tools

Choose to use the corresponding post training quantization tool according to the hardware platform deployed for model inference.

| Post Training Quantization Tools | Quantization Method Supported | Inference Hardware Platform Supported | Quantization Model Deployment |
| --- | --- | --- | --- |
| [MindSpore Post Training Quantization Tools](https://www.mindspore.cn/lite/docs/en/r1.5/use/post_training_quantization.html) | Weights Quantization <br> Full Quantization | CPU | [Inference on edge device](https://www.mindspore.cn/lite/docs/en/r1.5/use/runtime.html) |
| Ascend Model Compression Tool | Full Quantization | Ascend 310 AI Processor | [Inference on Ascend 310 AI Processor](https://www.mindspore.cn/docs/programming_guide/en/r1.5/multi_platform_inference_ascend_310.html) |
