# AvgPool2d功能差异

PyTorch: 对输入数据的H与W维执行平均池化。使用上，仅需指定池化后数据H和W维的期望shape即可。无需用户手工计算并指定`kernel_size`、`stride`等。

MindSpore：需用户手工计算并指定`kernel_size`、`stride`等。
