# ReduceMean功能差异

PyTorch: 对输入做自适应的平均池化，算法内部根据指定的输出大小计算出对应大小的结果。仅在输出为1*1时和MindSpore的ReduceMean一致。

MindSpore：计算指定维度数据的平均值。
