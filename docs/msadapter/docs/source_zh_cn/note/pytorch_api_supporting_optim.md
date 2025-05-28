# torch.optim

## Base class

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.optim.Optimizer](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Optimizer.html)|Not Support|N/A|
|[Optimizer.add_param_group](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Optimizer.add_param_group.html)|Not Support|N/A|
|[Optimizer.load_state_dict](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Optimizer.load_state_dict.html)|Not Support|N/A|
|[Optimizer.state_dict](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Optimizer.state_dict.html)|Not Support|N/A|
|[Optimizer.step](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Optimizer.step.html)|Not Support|N/A|
|[Optimizer.zero_grad](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Optimizer.zero_grad.html)|Not Support|N/A|

## Algorithms

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.optim.Adadelta](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Adadelta.html)|Not Support|N/A|
|[torch.optim.Adagrad](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Adagrad.html)|Not Support|N/A|
|[torch.optim.Adam](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Adam.html)|Beta|入参不支持foreach、capturable、differentiable、fused；支持数据类型：bf16、fp16、fp32|
|[torch.optim.AdamW](https://docs.pytorch.org/docs/2.1/generated/torch.optim.AdamW.html)|Beta|入参不支持foreach、capturable、differentiable、fused；参数weight_decay的默认值为1e-2，torch的默认值为0，支持数据类型：bf16、fp16、fp32|
|[torch.optim.SparseAdam](https://docs.pytorch.org/docs/2.1/generated/torch.optim.SparseAdam.html)|Not Support|N/A|
|[torch.optim.Adamax](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Adamax.html)|Not Support|N/A|
|[torch.optim.ASGD](https://docs.pytorch.org/docs/2.1/generated/torch.optim.ASGD.html)|Not Support|N/A|
|[torch.optim.LBFGS](https://docs.pytorch.org/docs/2.1/generated/torch.optim.LBFGS.html)|Not Support|N/A|
|[torch.optim.NAdam](https://docs.pytorch.org/docs/2.1/generated/torch.optim.NAdam.html)|Not Support|N/A|
|[torch.optim.RAdam](https://docs.pytorch.org/docs/2.1/generated/torch.optim.RAdam.html)|Not Support|N/A|
|[torch.optim.RMSprop](https://docs.pytorch.org/docs/2.1/generated/torch.optim.RMSprop.html)|Not Support|N/A|
|[torch.optim.Rprop](https://docs.pytorch.org/docs/2.1/generated/torch.optim.Rprop.html)|Not Support|N/A|
|[torch.optim.SGD](https://docs.pytorch.org/docs/2.1/generated/torch.optim.SGD.html)|Beta|入参不支持foreach、differentiable；参数lr具有默认值1e-3，torch无默认值；支持数据类型：bf16、fp16、fp32|

## How to adjust learning rate

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.optim.lr_scheduler.LambdaLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.LambdaLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.MultiplicativeLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.MultiplicativeLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.StepLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.StepLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.MultiStepLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.MultiStepLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.ConstantLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.ConstantLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.LinearLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.LinearLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.ExponentialLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.ExponentialLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.PolynomialLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.PolynomialLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.CosineAnnealingLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.ChainedScheduler](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.ChainedScheduler.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.SequentialLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.SequentialLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.ReduceLROnPlateau](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.CyclicLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.CyclicLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.OneCycleLR](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.OneCycleLR.html)|Not Support|N/A|
|[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts](https://docs.pytorch.org/docs/2.1/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)|Not Support|N/A|
