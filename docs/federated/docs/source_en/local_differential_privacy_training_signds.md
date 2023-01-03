# Horizontal FL-Local Differential Privacy SignDS training

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/federated/docs/source_en/local_differential_privacy_training_signds.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Privacy Protection Background

Federated learning enables the client user to participate in global model training without uploading the original dataset by allowing the participant to upload only the new model after local training or update the update information of the model, breaking through the data silos. This common scenario of federated learning corresponds to the default scheme in the MindSpore federated learning framework, where the `encrypt_type` switch defaults to `not_encrypt` when starting the `server`. The `installation and deployment` and `application practices` in the federated learning tutorial both use this approach by default, which is a common federated seeking averaging scheme without any privacy-protecting treatment such as cryptographic perturbation. For the convenience of description, `not_encrypt' is used below to refer specifically to this default scheme.

This federated learning scheme is not free from privacy leakage, using the above `not_encrypt` scheme for training. The Server receives the local training model uploaded by the Client, which can still reconstruct the user training data through some attack methods [1], thus leaking user privacy, so the `not_encrypt` scheme needs to further increase the user privacy protection mechanism.

The global model `oldModel` received by the Client in each round of federated learning is issued by the Server, which does not involve user privacy issues. However, the local model `newModel` obtained by each Client after several epochs of local training fits its local privacy data, so the privacy protection focuses on the weight difference between the two `newModel`-`oldModel`=`update`.

The `DP_ENCRYPT` differential noise scheme already implemented in the MindSpore Federated framework achieves privacy preservation by iteratively perturbing Gaussian random noise to `update`. However, as the dimensionality of the model increases, the increase in the `update` paradigm will increase the noise, thus requiring more Clients to participate in the same round of aggregation to neutralize the noise impact, otherwise the convergence and accuracy of the model will be reduced. If the noise is set too small, although the convergence and accuracy are close to the performance of the `not_encrypt` scheme, the privacy protection is not strong enough. Also each Client needs to send the perturbed model, and as the model increases, the communication overhead increases. We expect the Client represented by the cell phone to achieve convergence of the global model with as little communication overhead as possible.

## Algorithm Flow Introduction

SignDS [2] is the abbreviation of Sign Dimension Select, and the processing object is the `update` of Client. Preparation: each layer of Tensor of `update` is flattened and expanded into a one-dimensional vector, connected together, and the number of splicing vector dimensions is noted as $d$.

One sentence summarizes the algorithm: **Select $h(h<d)$ dimensions of `update`, replace the original update value of the selected dimension with the sign value (sign value: plus or minus 1), and replace the unselected ones with 0.**

Here is an example: there are 3 clients Client1, 2, 3, whose `update` is a $d=8$-dimensional vector after flattening and expanding, and the Server calculates the `avg` of these 3 clients Client and updates the global model with the value, that is, completes a round of federated learning.

| Client | d_1  | d_2  | d_3  | d_4  | d_5  | d_6  | d_7  |  d_8  |
| :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: |
|   1    | 0.4  | 0.1  | -0.2 | 0.3  | 0.5  | 0.1  | -0.2 | -0.3  |
|   2    | 0.5  | 0.2  |  0   | 0.1  | 0.3  | 0.2  | -0.1 | -0.2  |
|   3    | 0.3  | 0.1  | -0.1 | 0.5  | 0.2  | 0.3  |  0   |  0.1  |
|  avg   | 0.4  | 0.13 | -0.1 | 0.3  | 0.33 | 0.2  | -0.1 | -0.13 |

The dimension with higher importance should be selected, and the importance measure is the size of the **fetching value**, and the update needs to be sorted. update takes positive and negative values to represent different update directions, so in each round of federated learning, the sign values of Client each have **0.5 probability** of taking `1` or `-1`. If sign=1, the largest $k$ number of `update` dimensions are noted as the `topk` set and the remaining ones are noted as the `non-topk` set. If sign=-1, the smallest $k$ number of ones are noted as the `topk` set.

If the Server specifies `h`, the total number of selected dimensions, the Client will directly use this value, otherwise each Client will locally calculate the optimal output dimension `h`.

The SignDS algorithm outputs the number of dimensions (denoted as $v$) that should be selected from the `topk` set and the `non-topk` set, as in the example in the table below, where the two sets pick a total of dimensions h=3.

Client selects dimensions uniformly and randomly according to the number of dimensions output by the SignDS algorithm, sends the dimension number and sign value to the Server. If the dimension number is output in the order of picking from `topk` first and then from `non-topk`, the dimension number list `index` needs to be shuffled and disordered. The following table shows the information finally transferred from each Client of this algorithm to the Server.

| Client | index | sign |
| :----: | :---: | :--: |
|   1    | 1,5,8 |  1   |
|   2    | 2,3,4 |  -1  |
|   3    | 3,6,7 |  1   |

Server constructs `update` with privacy protection based on the dimension serial number and sign value uploaded by each client Client, aggregates and averages all `updates` and updates the current `oldModel` to complete a round of federated learning.

| Client |  d_1  |  d_2   |  d_3   |  d_4   |  d_5  |  d_6  |  d_7  |  d_8  |
| :----: | :---: | :----: | :----: | :----: | :---: | :---: | :---: | :---: |
|   1    | **1** |   0    |   0    |   0    | **1** |   0   |   0   | **1** |
|   2    |   0   | **-1** | **-1** | **-1** |   0   |   0   |   0   |   0   |
|   3    |   0   |   0    | **1**  |   0    |   0   | **1** | **1** |   0   |
|  avg   |  1/3  |  -1/3  |   0    |  -1/3  |  1/3  |  1/3  |  1/3  |  1/3  |

The optimized SignDS scheme has realized that the device-side client only uploads a list of dimension numbers of int type and a random Sign value of boolean type outputted by the algorithm to the cloud side, which significantly reduces the communication overhead compared to the common scenario of uploading thousands of float-level complete model weights or gradients. From the perspective of the actual reconstruction attack, the cloud side only gets the dimension serial number and a Sign value representing the direction of gradient update, and the attack is more difficult to achieve. The cloud side receives the dimension serial number and Sign value from the device side, and has to simulate the reconstruction of the original user weight, i.e., using `sign_global_lr` and Sign value, the latter representing the updated direction and the former representing the step size, which is where the accuracy of the scheme is lost. The cloud side can only be reconstructed to simulate each client **partial** gradient update. The number is equal to the number of serial numbers. Because the dimension selection is all random, the more the number of client users involved in aggregation, and the more model weights will be activated. If the reconstructed `update` is mostly focused on a certain position, it means that the real weight of that position is more updated, and vice versa, it means that the original update of that position is less updated. By reconstructing `update` and adding the initial model weights in this round, the cloud side can aggregate and update the final model in this round.

## Privacy Protection Certificate

The differential privacy noise scheme achieves privacy protection by adding noise so that the attacker cannot determine the original information, while the differential privacy SignDS scheme activates partial dimensions and replaces the original value with the sign value, which largely protects user privacy. Further, using the differential privacy index mechanism makes it impossible for an attacker to confirm whether the activated dimensions are significant (from the `topk` set) and whether the number of dimensions from `topk` in the output dimensions exceeds a given threshold.

For any two updates $\Delta$ and $\Delta'$ of each Client, the set of `topk` dimensions is $S_{topk}$ , ${S'}_{topk}$ , respectively. The set of any possible output dimensions of the algorithm is ${J}\in {\mathcal{J}}$ . Note that $\nu=|{S}_ {topk}\cap {J}|$ , $\nu'=|{S'}_{topk}\cap {J}|$ is the number of intersections of ${J}$ and `topk` sets, and the algorithm such that the following inequality holds:

$$
\frac{{Pr}[{J}|\Delta]}{{Pr}[{J}|\Delta']}=\frac{{Pr}[{J}|{S}_{topk}]}{{Pr}[{J}|{S'}_{topk}]}=\frac{\frac{{exp}(\frac{\epsilon}{\phi_u}\cdot u({S}_{topk},{J}))}{\sum_{{J'}\in {\mathcal{J}}}{exp}(\frac{\epsilon}{\phi_u}\cdot u({S}_{topk}, {J'}))}}{\frac{{exp}(\frac{\epsilon}{\phi_u}\cdot u({S'}_{topk}, {J}))}{\sum_{ {J'}\in {\mathcal{J}}}{exp}(\frac{\epsilon}{\phi_u}\cdot u( {S'}_{topk},{J'}))}}=\frac{\frac{{exp}(\epsilon\cdot \unicode{x1D7D9}(\nu \geq \nu_{th}))}{\sum_{\tau=0}^{\tau=\nu_{th}-1}\omega_{\tau} + \sum_{\tau=\nu_{th}}^{\tau=h}\omega_{\tau}\cdot {exp}(\epsilon)}}{\frac{ {exp}(\epsilon\cdot \unicode{x1D7D9}(\nu' \geq\nu_{th}))}{\sum_{\tau=0}^{\tau=\nu_{th}-1}\omega_{\tau}+\sum_{\tau=\nu_{th}}^{\tau=h}\omega_{\tau}\cdot {exp}(\epsilon)}}\\= \frac{{exp}(\epsilon\cdot \unicode{x1D7D9} (\nu \geq \nu_{th}))}{ {exp}(\epsilon\cdot \unicode{x1D7D9} (\nu' \geq \nu_{th}))} \leq \frac{{exp}(\epsilon\cdot 1)}{{exp}(\epsilon\cdot 0)} = {exp}(\epsilon),
$$

It is proved that the algorithm satisfies local differential privacy.

## Preparation

To use the algorithm, one first needs to successfully complete the training aggregation process for either cross-device federated scenario. [Implementing an Image Classification Application of Cross-device Federated Learning (x86)](https://www.mindspore.cn/federated/docs/en/r2.0.0-alpha/image_classification_application.html) describes the preparation work such as datasets, network models, and simulations to initiate the process of multi-client participation in federated learning in detail.

## Algorithm Opening Script

Local differential privacy SignDS training currently only supports cross-device federated learning scenarios. The opening method needs to change the following parameter configuration in the yaml file when opening the cloud-side service. The complete cloud-side opening script can be referred to the cloud-side deployment, and the relevant parameter configuration for opening this algorithm is given here. Taking LeNet task as an example, the yaml related configuration is as follows:

```python
encrypt:
  encrypt_type: SIGNDS
  ...
  signds:
    sign_k: 0.01
    sign_eps: 100
    sign_thr_ratio: 0.6
    sign_global_lr: 0.1
    sign_dim_out: 0
```

For the detailed example, refer to [Implementing an Image Classification Application of Cross-device Federated Learning (x86)](https://www.mindspore.cn/federated/docs/en/r2.0.0-alpha/image_classification_application.html). The cloud-side code implementation gives the definition domain of each parameter. If it is not in the definition domain, Server will report an error prompting the definition domain. The following parameter changes are subject to keeping the remaining 4 parameters unchanged.

- `sign_k`: (0,0.25], k*inputDim>50. default=0.01. `inputDim` is the pulling length of the model or update. If not satisfied, there is a device-side warning. Sort update, and the `topk` set is composed of the first k (%) of it. Decreasing k means to pick from more important dimensions with greater probability. The output will have fewer dimensions, but the dimensions are more important and the change in convergence cannot be determined. The user needs to observe the sparsity of model update to determine the value. When it is quite sparse (update has many zeros), it should be taken smaller.
- `sign_eps`: (0,100], default=100. Privacy-preserving budget. The number sequence symbol is $\epsilon$, abbreviated as eps. When eps decreases, the probability of picking unimportant dimensions increases. When privacy protection is enhanced, output dimensions decrease, the percentage remains the same, and precision decreases.
- `sign_thr_ratio`: [0.5,1], default=0.6. The dimension from `topk` in the activation dimension is occupied threshold lower bound. Increasing will reduce the output dimension, but the proportion of output dimensions from `topk` will increase. When the value is increased excessively, more from `topk` is required in the output, and the total output dimension can only be reduced to meet the requirement, and the accuracy decreases when the number of clients is not large enough.
- `sign_global_lr`: (0,), default=1. This value is multiplied by sign instead of update, which directly affects the convergence speed and accuracy. Moderately increasing this value will improve the convergence speed, but it may make the model oscillate and the gradient explode. If more epochs are run locally per client and the learning rate used for local training is increased, the value needs to be increased accordingly. If the number of clients involved in the aggregation increases, the value also needs to be increased, because the value needs to be aggregated and then divided by the number of users when reconstruction. The result will remain the same only if the value is increased.
- `sign_dim_out`: [0,50], default=0. If a non-zero value is given, the client side uses the value directly, increasing the value to output more dimensions, but the proportion of dimensions from `topk` will decrease. If it is 0, the client user has to calculate the optimal output parameters. If eps is not large enough, and the value is increased, many `non-topk` insignificant dimensions will be output leading to affect the mode convergence and accuracy decrease. When eps is large enough, increasing the value will allow important dimension information of more users to leave the local area and improve the accuracy.

## LeNet Experiment results

Use 100 client datasets of `3500_clients_bin`, 200 iterations of federated aggregation. 20 epochs run locally per client, and using learning rate of device-side local training is 0.01. The related parameter of SignDS is `k=0.01, eps=100, ratio=0.6, lr=4, out=0`, and the final accuracy is 66.5% for all users and 69% for the common federated scenario without encryption. In the unencrypted scenario, the length of the data uploaded to the cloud side at the end of training on the device side is 266,084, but the length of the data uploaded by SignDS is only 656.

## References

[1] Ligeng Zhu, Zhijian Liu, and Song Han. [Deep Leakage from Gradients](http://arxiv.org/pdf/1906.08935.pdf). NeurIPS, 2019.

[2] Xue Jiang, Xuebing Zhou, and Jens Grossklags. "SignDS-FL: Local Differentially-Private Federated Learning with Sign-based Dimension Selection." ACM Transactions on Intelligent Systems and Technology, 2022.