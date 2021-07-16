﻿# FAQ

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindarmour/faq/source_en/faq.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

<font size=3>**Q: What should I do when FastGradientSignMethod does not specify loss_fn, it reports an error: `Function construct_wrapper, the number of parameters of this function is 9, but the number of provided arguments is 10.`**</font>

A: In the `class mindarmour.adv_robustness.attacks.FastGradientSignMethod(network, eps=0.07, alpha=None, bounds=(0.0,1.0), is_targeted=False, loss_fn=None)`, if the incoming `network` is not in the form of `WithLossCell`, you need to specify `Loss_fn`, and the `WithLossCell` operation will be automatically executed by the execution process, otherwise there will be an error that the number of input parameters and the number of parameters required by the function are inconsistent.
