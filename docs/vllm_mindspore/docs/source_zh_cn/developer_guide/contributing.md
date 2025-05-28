# 贡献指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/developer_guide/contributing.md)

## 贡献者许可协议

向MindSpore社区提交代码之前，您需要签署《贡献者许可协议（CLA）》。
个人贡献者请参见[ICLA在线文件](https://www.mindspore.cn/icla)。

## 快速入门

- 在[Gitee](https://gitee.com/mindspore/vllm-mindspore)上fork代码仓。
- 参见[README.md](https://gitee.com/mindspore/vllm-mindspore/blob/master/README.md)和安装页面了解项目信息和构建说明。

## 增加新模型

若希望将一个新模型合入vLLM MindSpore代码仓库，需要注意几点：

- **文件格式及位置要遵循规范。** 模型代码文件统一放置于`vllm_mindspore/model_executor`文件夹下，请根据不同模型将代码文件放置于对应的文件夹下。
- **模型基于MindSpore接口实现，支持jit静态图方式执行。** vLLM MindSpore中的模型定义实现需基于MindSpore接口实现。由于MindSpore静态图模式执行性能有优势，因此模型需支持@jit静态图方式执行。详细可参考[Qwen2.5](https://gitee.com/mindspore/vllm-mindspore/blob/master/vllm_mindspore/model_executor/models/qwen2.py)模型定义实现。
- **将新模型在vLLM MindSpore代码中进行注册。** 模型结构定义实现后，需要将该模型注册到vLLM MindSpore中，注册文件位于'vllm_mindspore/model_executor/models/registry.py'中，请将模型注册到`_NATIVE_MODELS`。
- **编写单元测试。** 新增的模型需同步提交单元测试用例，用例编写请参考[Qwen2.5模型用例](https://gitee.com/mindspore/vllm-mindspore/blob/master/tests/st/python/test_vllm_qwen_7b.py)。

## 贡献流程

### 代码风格

请遵循此风格，以便社区代码的审查、维护和开发。

- **编码指南：** 使用vLLM社区代码检查工具：yapf、codespell、ruff、isort和mypy。更多信息可参考[检查工具链使用说明](https://gitee.com/mindspore/vllm-mindspore/blob/master/codecheck_toolkits/README.md)。

- **单元测试指南：** vLLM MindSpore使用Python单元测试框架[pytest](http://www.pytest.org/en/latest/)。注释名称需反映测试用例的设计意图。

- **重构指南：** 我们鼓励开发人员重构我们的代码，以消除[代码坏味道](https://zh.wikipedia.org/wiki/%E4%BB%A3%E7%A0%81%E5%BC%82%E5%91%B3)。所有代码都要符合编码风格和测试风格，重构代码也不例外。

### Fork-Pull开发模型

- **Fork vLLM MindSpore代码仓：** 在提交代码至vLLM MindSpore项目之前，请确保已fork此项目到您自己的代码仓。vLLM MindSpore代码仓和您自己的代码仓之间可能会并行开发，请注意它们之间的一致性。

- **克隆远程代码仓：** 如果您想将代码下载到本地计算机，最好使用git方法：

    ```shell
    # 在Gitee上：
    git clone https://gitee.com/{insert_your_forked_repo}/vllm-mindspore.git
    git remote add upstream https://gitee.com/mindspore/vllm-mindspore.git
    ```

- **本地开发代码：** 为避免分支不一致，建议切换到新分支：

    ```shell
    git checkout -b {新分支名称} origin/master
    ```

    以master分支为例，如果需要创建版本分支和下游开发分支，请先修复上游的bug后再更改代码。

- **将代码推送到远程代码仓：** 更新代码后，以正式的方式推送更新：

    ```shell
    git add .
    git status # 查看更新状态。
    git commit -m "你的commit标题"
    git commit -s --amend # 添加commit的具体描述。
    git push origin {新分支名称}
    ```

- **将请求拉取到vLLM MindSpore代码仓：** 在最后一步中，您需要在新分支和vLLM MindSpore主分支之间拉取比较请求然后创建PR。提交PR提交后，需要在评论中通过`/retest`手动触发门禁检查，进行构建测试。PR应该尽快合并到上游master分支中，以降低合并的风险。

### 报告Issue

发现问题后，建议以报告[issue](https://gitee.com/mindspore/vllm-mindspore/issues)的方式为项目作出贡献。错误报告应尽量书写规范，内容详尽。

报告issue时，请参考以下格式：

- 说明您使用的环境版本（vLLM MindSpore、MindFormers、MindSpore、OS、Python等）;
- 说明是错误报告还是功能需求；
- 说明issue类型，添加标签可以在issue板上突出显示该issue;
- 问题是什么；
- 期望如何处理；
- 如何复现（尽可能精确具体地描述）；
- 给审核员的特别说明。

**Issue咨询：**

- **解决issue时，请先评论**，告知他人由您来负责解决该issue。
- **对于长时间未关闭的issue**，建议贡献者在解决该issue之前进行预先检查。
- **如您自行解决了自己报告的issue**，仍需在关闭该issue之前告知他人。

### 提交PR

- 如果是需要大量设计细节的新功能，还应提交设计方案。
- 经issue讨论和设计方案评审达成共识后，在已fork的代码仓开发，并提交PR。
- 任何PR至少需要位2位审批人的LGTM标签。请注意，审批人不允许在自己的PR上添加LGTM标签。
- 经充分讨论后，根据讨论的结果合并、放弃或拒绝PR。

**PR咨询：**

- 避免不相关的更改。
- 确保您的commit历史记录有序。
- 确保您的分支与主分支始终一致。
- 用于修复错误的PR中，确保已关联所有相关问题。

最后，感谢您对为vLLM MindSpore项目做出贡献的兴趣，我们欢迎并重视任何形式的贡献与合作。
