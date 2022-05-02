# Pairwise encryption training

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/pairwise_encryption_training.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

During federated learning, user data is used only for local device training and does not need to be uploaded to the central server. This prevents personal data leakage.
However, in the conventional federated learning framework, models are migrated to the cloud in plaintext. There is still a risk of indirect disclosure of user privacy.
After obtaining the plaintext model uploaded by a user, the attacker can restore the user's personal training data through attacks such as reconstruction and model inversion. As a result, user privacy is disclosed.

As a federated learning framework, MindSpore Federated provides secure aggregation algorithms based on local secure multi-party computation (MPC).
Secret noise addition is performed on local models before they are migrated to the cloud. On the premise of ensuring the model availability, the problem of privacy leakage and model theft in horizontal federated learning are solved.

## Principles

Although the LDP technology can properly protect user data privacy, when there are a relatively small quantity of participating clients or a Gaussian noise amplitude is relatively large, the model accuracy is greatly affected.
To meet both model protection and model convergence requirements, we provide the MPC-based secure aggregation solution.

In this training mode, assuming that the participating client set is $U$, for any client $u$ and $v$, they negotiate a pair of random perturbations $p_{uv}$ and $p_{vu}$, which meet the following condition:

$$
p_{uv}=\begin{cases} -p_{vu}, &u{\neq}v\\\\ 0, &u=v \end{cases}
$$

Therefore, each client $u$ adds the perturbation negotiated with other users to the original model weight $x_u$ before uploading the model to the server:

$$
x_{encrypt}=x_u+\sum\limits_{v{\in}U}p_{uv}
$$

Therefore, the server aggregation result $\overline{x}$ is as follows:

$$
\begin{align}
\overline{x}&=\sum\limits_{u{\in}U}(x_{u}+\sum\limits_{v{\in}U}p_{uv})\\\\
&=\sum\limits_{u{\in}U}x_{u}+\sum\limits_{u{\in}U}\sum\limits_{v{\in}U}p_{uv}\\\\
&=\sum\limits_{u{\in}U}x_{u}
\end{align}
$$

The preceding process describes only the main idea of the aggregation algorithm. The MPC-based aggregation solution is accuracy-lossless but increases the number of communication rounds.
If you are interested in the specific steps of the algorithm, refer to the paper[1].

## Usage

### Cross device scenario

Enabling pairwise encryption training is simple. You only need to set `encrypt_type='PW_ENCRYPT'` in `context.set_fl_context()`.

In addition, most of the workers participating in the training are unstable edge computing nodes such as mobile phones, so the problems of dropping the line and secret key reconstruction should be considered. Related parameters are `share_secrets_ratio`, `reconstruct_secrets_threshold`, and `cipher_time_window`.

`share_client_ratio` indicates the ratio of the number of clients participating in key fragment sharing to the number of clients participating in federated learning. The value must be less than or equal to 1.

`reconstruct_secrets_threshold` indicates the number of clients that participate in key fragment reconstruction. The value must be less than the number of clients that participate in key fragment sharing.

To ensure system security, the `reconstruct_secrets_threshold` must be greater than half of the number of federated learning clients when the server and client are not colluded.
When the server and client are colluded, the value of `reconstruct_secrets_threshold` must be greater than two thirds of the number of federated learning clients.

`cipher_time_window` indicates the duration limit of each communication round for secure aggregation. It is used to ensure that the server can start a new round of iteration when some clients are offline.
It should be noted that only `server_num=1` is supported for current PW_ENCRYPT mode.

### Cross silo scenario

In cross silo scenario, you only need to set `encrypt_type='STABLE_PW_ENCRYPT'` in `context.set_fl_context()` for both server startup script and client startup script.

Different from cross silo scenario, all of the workers are stable computing nodes in cross silo scenario. You only need to set the parameter `cipher_time_window`.

## References

[1] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, et al. [Practical Secure Aggregationfor Privacy-Preserving Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3133956.3133982). NeurIPS, 2016.
