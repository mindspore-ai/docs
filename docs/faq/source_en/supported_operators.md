# Supported Operators

`Ascend` `GPU` `CPU` `Environmental Setup` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/tree/r1.0/docs/faq/source_en/supported_operators.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q: What can I do if the LSTM example on the official website cannot run on Ascend?

A: Currently, the LSTM runs only on a GPU or CPU and does not support the hardware environment. You can click [here](https://www.mindspore.cn/doc/note/en/r1.0/operator_list_ms.html) to view the supported operators.

<br/>

Q: When conv2d is set to (3,10), Tensor[2,2,10,10] and it runs on Ascend on ModelArts, the error message `FM_W+pad_left+pad_right-KW>=strideW` is displayed. However, no error message is displayed when it runs on a CPU. What should I do?

A: This is a TBE operator restriction that the width of x must be greater than that of the kernel. The CPU does not have this operator restriction. Therefore, no error is reported.