# FAQ

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_zh_cn/faq.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: FastGradientSignMethod未指定loss_fn时报错`Function construct_wrapper, The number of parameters of this function is 9, but the number of provided arguments is 10.`怎么办？**</font>

A: class mindarmour.adv_robustness.attacks.FastGradientSignMethod(network, eps=0.07, alpha=None, bounds=(0.0,1.0), is_targeted=False, loss_fn=None)中，如果传入的network不是WithLossCell的形式，需要指定Loss_fn，由执行过程自动执行WithLossCell操作，否则会出现输入参数数量和函数所需参数数量不一致的错误。
