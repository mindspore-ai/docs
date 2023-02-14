# mindspore.nn API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.nn.CTCLoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.CTCLoss.html#mindspore.nn.CTCLoss)|Changed|Calculates the CTC (Connectionist Temporal Classification) loss.|r2.0.0-alpha: Ascend/CPU => master: Ascend/GPU/CPU|Loss Function
[mindspore.nn.MultiLabelSoftMarginLoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MultiLabelSoftMarginLoss.html#mindspore.nn.MultiLabelSoftMarginLoss)|New|Calculates the MultiLabelSoftMarginLoss.|master: Ascend/GPU/CPU|Loss Function
[mindspore.nn.PoissonNLLLoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.PoissonNLLLoss.html#mindspore.nn.PoissonNLLLoss)|New|Poisson negative log likelihood loss.|r2.0.0-alpha: master: Ascend/GPU/CPU|Loss Function
[mindspore.nn.SoftMarginLoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.SoftMarginLoss.html#mindspore.nn.SoftMarginLoss)|Changed|A loss class for two-class classification problems.|r2.0.0-alpha: Ascend/GPU => master: Ascend|Loss Function
[mindspore.nn.TripletMarginLoss](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.TripletMarginLoss.html#mindspore.nn.TripletMarginLoss)|New|TripletMarginLoss operation.|master: GPU|Loss Function
[mindspore.nn.ProximalAdagrad](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.ProximalAdagrad.html#mindspore.nn.ProximalAdagrad)|Changed|Implements the ProximalAdagrad algorithm.|r2.0.0-alpha: Ascend/GPU/CPU => master: Ascend/GPU|Optimizer
[mindspore.nn.ReflectionPad3d](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.ReflectionPad3d.html#mindspore.nn.ReflectionPad3d)|New|Using a given padding to do reflection pad the given tensor.|master: Ascend/GPU/CPU|Padding Layer
[mindspore.nn.Identity](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Identity.html#mindspore.nn.Identity)|New|Returns a Tensor with the same shape and contents as input.|master: Ascend/GPU/CPU|Tools
[mindspore.nn.Unflatten](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Unflatten.html#mindspore.nn.Unflatten)|New|Summary:|master: Ascend/GPU/CPU|Tools
[mindspore.nn.WithGradCell](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/nn/mindspore.nn.WithGradCell.html#mindspore.nn.WithGradCell)|Deleted|Cell that returns the gradients.|Ascend/GPU/CPU|Wrapper Layer
