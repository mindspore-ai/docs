# 局部差分隐私SignDS训练

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/local_differential_privacy_training_signds.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 隐私保护背景

联邦学习通过让参与方只上传本地训练后的新模型或更新模型的update信息，实现了client用户不上传原始数据集就能参与全局模型训练的目的，打通了数据孤岛。这种普通场景的联邦学习对应MindSpore联邦学习框架中的默认方案（[云侧部署](https://www.mindspore.cn/federated/docs/zh-CN/master/deploy_federated_server.html#id5)启动`server`时，`encrypt_type`开关默认为`not_encrypt`，联邦学习教程中的`安装部署`与`应用实践`都默认使用这种方式），是没有任何加密扰动等保护隐私处理的普通联邦求均方案，为方便描述，下文以`not_encrypt`来特指这种默认方案。

这种联邦学习方案并不是毫无隐私泄漏的，使用上述`not_encrypt`方案进行训练，服务端server收到用户client的训练后模型，仍可通过一些攻击方法[1]重构用户训练数据，从而泄露用户隐私，所以`not_encrypt`方案需要进一步进行用户隐私保护。

联邦学习每轮client接收的当前模型`oldModel`都是server下发的，不涉及到用户隐私问题。但每个client本地跑若干epoch后得到的新模型`newModel`拟合了该用户本地隐私数据，所以更具体的是要保护权重差值`newModel`-`oldModel`=`update`的隐私。

MindSpore联邦学习框架中已实现的`DP_ENCRYPT`差分噪声方案通过在用户`update`加高斯随机噪声进行扰动，从而保护隐私。但随着模型维度增大，`update`范数增大会使噪声增大，同时需要较多的client用户参与同一轮聚合去中和噪声，否则模型收敛性和精度会降低。如果设置的噪声过小，虽然收敛性和精度与`not_encrypt`方案性能接近，但隐私保护力度不够。同时每个client用户都需要发送扰动后的模型，随着模型增大，通信开销也要随之增大。我们希望端侧手机用户尽可能少的去通信就可实现全局模型的收敛。

## 算法流程介绍

SignDS[2]是Sign Dimension Select的缩写，处理对象是client用户的`update`。准备工作：把`update`的每一层Tensor拉平展开成一维向量，连接在一起，拼接向量维度数量记为$d$。

一句话概括算法：**选择`update`的$h(h<d)$个维度，用sign值（符号值：正负1）代替这些维度原始的update值，未选中的用0代替。**

下面举例来说明：现有3个client用户1，2，3，其`update`拉平展开后为$d=8$维向量，server计算这3个client用户的`avg`，并用该值更新全局模型，即完成一轮联邦学习。

| Client | d_1  | d_2  | d_3  | d_4  | d_5  | d_6  | d_7  |  d_8  |
| :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: |
|   1    | 0.4  | 0.1  | -0.2 | 0.3  | 0.5  | 0.1  | -0.2 | -0.3  |
|   2    | 0.5  | 0.2  |  0   | 0.1  | 0.3  | 0.2  | -0.1 | -0.2  |
|   3    | 0.3  | 0.1  | -0.1 | 0.5  | 0.2  | 0.3  |  0   |  0.1  |
|  avg   | 0.4  | 0.13 | -0.1 | 0.3  | 0.33 | 0.2  | -0.1 | -0.13 |

选择维度要挑重要的维度，重要性衡量是**值的大小**，需要对update进行排序。真实update值有正负，代表不同的更新方向，故每轮client用户的sign值各有**0.5的概率**取`1`或`-1`。如果sign=1，那么就把最大的$k$个`update`维度记为`topk`集合，剩余的记为`non-topk`集合；如果sign=-1，则把最小的$k$个记为`topk`集合。

如果server指定总共选择的维度数量`h`，client用户会直接使用该值，否则每个client用户会本地会计算出最优的输出维度`h`。

随后SignDS算法会输出应该从`topk`集合中挑多少维（数量记为$v$），从`non-topk`中挑多少维，下表格示例中，两个集合总共挑选维度$h=3$，

Client用户按照算法输出的指定维度去均匀随机挑选维度，将维度序号和sign值发到server即可，维度序号如果按照先从`topk`挑，再从`non-topk`挑的顺序输出，则需要对维度序号列表`index`进行洗牌打乱的操作，下表为该算法每个client用户最终要传到server端的信息：

| Client | index | sign |
| :----: | :---: | :--: |
|   1    | 1,5,8 |  1   |
|   2    | 2,3,4 |  -1  |
|   3    | 3,6,7 |  1   |

server根据每个client用户的维度和sign值构建隐私保护的`update`，对所有`update`进行聚合平均并更新当前`oldModel`即完成一轮联邦学习。

| Client |  d_1  |  d_2   |  d_3   |  d_4   |  d_5  |  d_6  |  d_7  |  d_8  |
| :----: | :---: | :----: | :----: | :----: | :---: | :---: | :---: | :---: |
|   1    | **1** |   0    |   0    |   0    | **1** |   0   |   0   | **1** |
|   2    |   0   | **-1** | **-1** | **-1** |   0   |   0   |   0   |   0   |
|   3    |   0   |   0    | **1**  |   0    |   0   | **1** | **1** |   0   |
|  avg   |  1/3  |  -1/3  |   0    |  -1/3  |  1/3  |  1/3  |  1/3  |  1/3  |

1.7版本中，优化后的SignDS方案已实现端侧client只上传算法输出的int类型维度序号列表和一个布尔类型的随机Sign值到云测，相比普通场景中上传数万float级别的完整模型权重或梯度，通讯开销显著降低。从实际重构攻击的角度来看，云测仅获得维度序号和代表梯度更新方向的一个Sign值，攻击更加难以实现。云测接收到端侧传来的维度序号和Sign值，要模拟重构出用户原始权重，即利用`sign_global_lr`和Sign值，后者代表更新的方向，前者代表步长，这也是该方案精度损失的地方。云测只能重构模拟出每个client**部分**梯度更新，数量等于序号数目，且因为维度选择都是随机的，所以参与聚合的client用户数量越多，激活的模型权重也会越多。如果重构出的`update`大多聚焦在某个位置，则说明该位置真实权重更新较大，反之说明原始该位置update更新较小。云测通过重构`update`再加上本轮初始模型权重，便可聚合更新此轮最终模型。

## 隐私保护证明

差分隐私噪声方案通过加噪的方式，让攻击者无法确定原始信息从而实现隐私保护效果；而差分隐私SignDS方案只激活部分维度，且用sign值代替原始值，很大程度上保护了用户隐私。进一步的，利用差分隐私指数机制让攻击者无法确认激活的维度是否是重要（来自`topk`集合），且无法确认输出维度中来自`topk`的维度数量是否超过给定阈值。

对于每个client用户的任意两个update $\Delta$ 和 $\Delta'$  ，它们的`topk`维度集合分别是  $S_{topk}$ ， ${S'}_{topk}$ ，该算法任意可能的输出维度集合是 ${J}\in {\mathcal{J}} $ ，记 $\nu=|{S}_{topk}\cap {J}|$ ,  $\nu'=|{S'}_{topk}\cap {J}|$  是 ${J}$ 和`topk` 集合交集的数量，算法使得以下不等式成立：

$$
\frac{{Pr}[{J}|\Delta]}{{Pr}[{J}|\Delta']}=\frac{{Pr}[{J}|{S}_{topk}]}{{Pr}[{J}|{S'}_{topk}]}=\frac{\frac{{exp}(\frac{\epsilon}{\phi_u}\cdot u({S}_{topk},{J}))}{\sum_{{J'}\in {\mathcal{J}}}{exp}(\frac{\epsilon}{\phi_u}\cdot u({S}_{topk}, {J'}))}}{\frac{{exp}(\frac{\epsilon}{\phi_u}\cdot u({S'}_{topk}, {J}))}{\sum_{ {J'}\in {\mathcal{J}}}{exp}(\frac{\epsilon}{\phi_u}\cdot u( {S'}_{topk},{J'}))}}=\frac{\frac{{exp}(\epsilon\cdot \unicode{x1D7D9}(\nu \geq \nu_{th}))}{\sum_{\tau=0}^{\tau=\nu_{th}-1}\omega_{\tau} + \sum_{\tau=\nu_{th}}^{\tau=h}\omega_{\tau}\cdot {exp}(\epsilon)}}{\frac{ {exp}(\epsilon\cdot \unicode{x1D7D9}(\nu' \geq\nu_{th}))}{\sum_{\tau=0}^{\tau=\nu_{th}-1}\omega_{\tau}+\sum_{\tau=\nu_{th}}^{\tau=h}\omega_{\tau}\cdot {exp}(\epsilon)}}\\= \frac{{exp}(\epsilon\cdot \unicode{x1D7D9} (\nu \geq \nu_{th}))}{ {exp}(\epsilon\cdot \unicode{x1D7D9} (\nu' \geq \nu_{th}))} \leq \frac{{exp}(\epsilon\cdot 1)}{{exp}(\epsilon\cdot 0)} = {exp}(\epsilon),
$$

证明该算法满足局部差分隐私。

## 准备工作

若要使用该算法，首先需要成功完成任一端云联邦场景的训练聚合过程，[实现一个端云联邦的图像分类应用(x86)](https://www.mindspore.cn/federated/docs/zh-CN/master/image_classification_application.html)和[实现一个情感分类应用(Android)](https://www.mindspore.cn/federated/docs/zh-CN/master/sentiment_classification_application.html)都详细介绍了包括数据集和网络模型等准备工作和模拟启动多客户端参与联邦学习的流程。

## 算法开启脚本

本地差分隐私SignDS训练目前只支持端云联邦学习场景。开启方式需要在启动云侧服务时，使用`context.set_fl_context()`设置`encrypt_type='SIGNDS'`即可，云侧完整启动脚本可参考云侧部署的[run_mobile_server.py脚本](https://gitee.com/mindspore/mindspore/blob/master/tests/st/fl/mobile/run_mobile_server.py)，这里给出启动该算法的相关参数配置。以LeNet任务为例，在确定参数配置后，用户需要在执行训练前调用`set_fl_context`接口，传入算法参数，调用方式如下：

```python
...
# 打开开关
parser.add_argument("--encrypt_type", type=str, default="SIGNDS")
# SIGNDS方案的参数设计
parser.add_argument("--sign_k", type=float, default=0.01)
parser.add_argument("--sign_eps", type=float, default=100)
parser.add_argument("--sign_thr_ratio", type=float, default=0.6)
parser.add_argument("--sign_global_lr", type=float, default=5)
parser.add_argument("--sign_dim_out", type=int, default=0)
...
sign_k = args.sign_k
sign_eps = args.sign_eps
sign_thr_ratio = args.sign_thr_ratio
sign_global_lr = args.sign_global_lr
sign_dim_out = args.sign_dim_out
ctx = {
    ...
    "sign_k": sign_k,
    "sign_eps": sign_eps,
    "sign_thr_ratio": sign_thr_ratio,
    "sign_global_lr": sign_global_lr,
    "sign_dim_out": sign_dim_out
}
context.set_fl_context(**fl_ctx)
...
```

以下是部分关键参数传入server脚本的示例：

```python
...
# 打开开关
parser.add_argument("--encrypt_type", type=str, default="SIGNDS")
# SIGNDS方案的参数设计
parser.add_argument("--sign_k", type=float, default=0.01)
parser.add_argument("--sign_eps", type=float, default=100)
parser.add_argument("--sign_thr_ratio", type=float, default=0.6)
parser.add_argument("--sign_global_lr", type=float, default=5)
parser.add_argument("--sign_dim_out", type=int, default=0)
if __name__ == "__main__":
    ...
    sign_k = args.sign_k
    sign_eps = args.sign_eps
    sign_thr_ratio = args.sign_thr_ratio
    sign_global_lr = args.sign_global_lr
    sign_dim_out = args.sign_dim_out
    ...
    for i in range(local_server_num):
        ...
        cmd_server += " --sign_k=" + str(sign_k)
        cmd_server += " --sign_eps=" + str(sign_eps)
        cmd_server += " --sign_thr_ratio=" + str(sign_thr_ratio)
        cmd_server += " --sign_global_lr=" + str(sign_global_lr)
        cmd_server += " --sign_dim_out=" + str(sign_dim_out)
        ...
```

云侧代码实现给出了各个参数的定义域，不在定义域内的，server会报错提示定义域。以下参数改动的前提是保持其余4个参数不变：

- `sign_k`：(0,0.25]，k*inputDim>50. default=0.01，`inputDim`是模型或update的拉平长度，若不满足，端侧警告。排序update，占比前k（%）的组成`topk`集合。减少k，则意味着要从更重要的维度中以较大概率挑选，输出的维度会减少，但维度更重要，无法确定收敛性的变化，用户需观察模型update稀疏度来确定该值，当比较稀疏时（update有很多0），则应取小一点。
- `sign_eps`：(0,100]，default=100。隐私保护预算，数序符号为$\epsilon$，简写为eps。eps减少，挑选不重要的维度概率会增大，隐私保护力度增强，输出维度减少，占比不变，精度降低。
- `sign_thr_ratio`：[0.5,1]，default=0.6。激活的维度中来自`topk`的维度占比阈值下界。增大会减少输出维度，但输出维度中来自`topk`的占比会增加，当过度增大该值，要求输出中更多的来自`topk`，为了满足要求只能减少总的输出维度，当client用户数量不够多时，精度下降。
- `sign_global_lr`：(0,)，default=1。该值乘上sign来代替update，直接影响收敛快慢与精度，适度增大该值会提高收敛速度，但有可能让模型震荡，梯度爆炸。如果每个client用户本地跑更多的epoch，且增大本地训练使用的学习率，那么需要相应提高该值；如果参与聚合的client用户数目增多，那么也需要提高该值，因为重构时需要把该值聚合再除以用户数目，只有增大该值，结果才保持不变。
- `sign_dim_out`：[0,50]，default=0。若给出非0值，client端直接使用该值，增大该值输出的维度增多，但来自`topk`的维度占比会减少；若为0，client用户要计算出最优的输出参数。eps不够大时，若增大该值，则会输出很多`non-topk`的不重要维度导致影响模型收敛，精度下降；当eps足够大时，增大该值会让更多的用户重要的维度信息离开本地，精度提升。

## LeNet实验结果

使用`3500_clients_bin`其中的100个client数据集，联邦聚合200个iteration，每个client本地运行20个epoch，端侧本地训练使用学习率为0.01，SignDS相关参数为`k=0.01,eps=100,ratio=0.6,lr=4,out=0`，最终所有用户的准确率为66.5%，不加密的普通联邦场景为69%。不加密场景中，端侧训练结束上传到云测的数据长度为266084，但SignDS上传的数据长度仅为656。

## 参考文献

[1] Ligeng Zhu, Zhijian Liu, and Song Han. [Deep Leakage from Gradients](http://arxiv.org/pdf/1906.08935.pdf). NeurIPS, 2019.

[2] Xue Jiang, Xuebing Zhou, and Jens Grossklags. "SignDS-FL: Local Differentially-Private Federated Learning with Sign-based Dimension Selection." ACM Transactions on Intelligent Systems and Technology, 2022.