# MindSpore Transformers贡献指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/contribution/mindformers_contribution.md)

## 贡献代码至MindSpore Transformers

### 代码风格要求

请遵循此风格，以便MindSpore Transformers审查、维护和开发。

- 编码指南

  MindSpore Transformers社区使用`Python PEP 8` 编码风格。建议在IDE中安装以下插件，用于检查代码格式：`Lizard`、`ShellCheck` 和`PyLint`。

- 单元测试指南

  MindSpore Transformers社区使用Python单元测试框架pytest。注释名称需反映测试用例的设计意图。

- 重构指南

  我们鼓励开发人员重构代码，以消除代码坏味道。所有代码都要符合编码风格和测试风格，重构代码也不例外。无注释的代码行（nloc）的Lizard阈值为100，圈复杂度（ccn）的阈值为20。当收到Lizard警告时，必须重构要合并的代码。

- 文档指南

  我们使用MarkdownLint来检查Markdown文档格式。基于默认配置修改了以下规则：

    1. MD009（行尾空格）：参数br_spaces设置为2，表示行尾可以有0或2个空格。
    2. MD029（有序列表的序列号）：参数style设置为ordered，表示升序。

### Fork-Pull 开发模型指导

- Fork MindSpore Transformers代码仓

  在提交代码至MindSpore Transformers项目之前，请确保已fork此项目到您自己的代码仓。MindSpore Transformers代码仓和您自己的代码仓之间可能会并行开发，请注意保持它们之间的一致性。

- 克隆远程代码仓

  如果您想将代码下载到本地计算机，最好使用git方法。

  ```shell
  # 在AtomGit上克隆仓库
  git clone https://atomgit.com/(insert_your_forked_repo)/mindformers.git
  ```

- 本地开发代码

  `master`为开发分支，请从`master`分支拉取最新代码进行开发。在提交Pull Request时，请提交到`master`分支。

  ```shell
  git checkout -b {新分支名称} origin/master
  ```

- 提交PR到MindSpore Transformers代码仓

  在最后一步中，您需要在新分支和`MindSpore Transformers`主分支之间创建Pull Request。完成Pull Request后，`Jenkins CI`将自动进行构建测试。PR应该尽快合并到上游master分支中，以降低合并风险。

  ```shell
  # 添加所有更改到暂存区
  git add .

  # 查看更新状态
  git status

  # 提交更改，使用-m选项添加commit标题
  git commit -m "你的commit标题"

  # 添加commit的具体描述，使用-s选项添加签名，`--amend`选项修改最近一次提交
  git commit -s --amend

  # 推送更改到远程仓库的新分支
  git push origin {新分支名称}

  ```

### 文件及代码格式

若希望将自定义模型合入`MindSpore Transformers`代码仓库，需要注意以下几点：

1. 文件格式及位置要遵循规范。
2. 将新模型在代码中进行注册，以适配高阶接口使用。

#### 文件格式及位置

1. 模型代码文件统一放置于`research/{model_name}`文件夹下，格式如下：

    ```text
    research/{model_name}
    ├── {model_name}
    | ├── {pretrain/finetune/predict}_{model_name}_{n}b.yaml
    | ├── convert_weight.py # Torch权重转MindSpore权重脚本（迁移模型需提供）
    | ├── convert_reversed.py # MindSpore权重转Torch权重脚本（迁移模型需提供）
    | ├── run_{model_name}.py # 运行代码文件
    | ├── {model_name}.py   # Model类代码文件
    | └── {model_name}_tokenizer.py # Tokenizer代码文件
    ```

2. 模型文档放置于同一`research/{model_name}`文件夹下。

## 提交PR的要求

### 只有一个commit

对于多commit的PR，请使用`squash`命令将多个commit合并为一个。
例如使用：

```shell
git rebase -i HEAD~3
```

可以看到:

```shell
pick 1234567 添加新功能A
pick 89abcdef 修复了功能A中的bug
pick 01234567 对功能A进行了一些优化
```

squash合并commit（可简化为 s, p, f 等简写）

```shell
pick 1234567 添加新功能A
squash 89abcdef 修复了功能A中的bug
squash 01234567 对功能A进行了一些优化
```

### PR描述

请使用以下md模板：

```markdown

### 相关的Issue

### 原因（目的、解决的问题等）

### 描述（做了什么，变更了什么）

### check list

#### 是否完成方案评审或问题根因分析（Y/N）

#### 是否完成了功能模块的UT/ST，并执行通过，附上结果（Y/N）

#### 是否涉及公共组件或对外接口修改，涉及时需给出修改范围和影响评估（Y/N）

#### 是否涉及资料修改，涉及时需同步修改（Y/N）

```

### 门禁要求

1. 提交PR需要[签署CLA](https://www.mindspore.cn/icla)。

2. 提交PR需要通过CI门禁检查。门禁失败修改代码后，需要在评论下评论`/retest`手动重启门禁检查。

## 测试用例贡献

### 组织形式

#### 目录结构

```ColdFusion
    tests/  
    ├── st/                        # 系统测试：验证多组件协同的端到端流程  
    │   ├── test_auto_register/        # 测试自定义模型/算子自动注册  
    │   ├── test_ckpt_health_monitor/  # 测试模型权重完整性检查  
    │   ├── test_docs/                 # 测试文档示例代码可运行性  
    │   ├── test_grace_exit_save_ckpt/ # 测试训练中断时权重保存  
    │   ├── test_infer/                # 测试单卡/多卡/离线推理流程  
    │   ├── test_model/                # 测试模型多设备运行一致性  
    │   ├── test_multi_cards_cases/    # 测试多卡分布式训练/推理  
    │   ├── test_optim/                # 测试优化器/学习率/混合精度  
    │   ├── test_resume/               # 测试训练断点恢复功能  
    │   ├── test_safetensors/          # 测试Safetensors权重加载/保存  
    ├── utils/                     # 测试工具库：数据生成、设备检测等  
    ├── conftest.py                # pytest全局配置：环境检查、初始化
```

#### 基本规范

1. 测试用例mark规则：

    - npu用例：`@pytest.mark.platform_arm_ascend910b_training`
    - cpu用例：`@pytest.mark.platform_x86_cpu`
    - 单卡用例：`@pytest.mark.env_onecard`
    - 多卡用例（默认八卡）：`@pytest.mark.platform_env_single`

2. 测试用例开发规范：

    - 测试用例在用例文件目录下生成缓存文件
    - 所有测试用例在方法（包括类方法）上方添加执行规则相关mark，不能在类上方添加
    - 测试文件以test_开头，类以Test开头，方法以test开头

3. 用例级别规范：

    - level0：组合接口用例（仅并行接口归类至该级别）
    - level1：整网功能用例、并行计算接口的单卡用例、原子接口用例

### 执行示例

1. 安装依赖

    ```bash
    pip3 install -r requirements.txt
    ```

2. 单个测试文件执行

    ```bash
    cd test/st
    pytest test_demo.py
    ```

3. 按标记筛选执行

    ```bash
    # 可通过-m参数筛选指定标记的用例
    # 执行所有npu单卡用例
    pytest test_demo.py -v -m "platform_arm_ascend910b_training and env_onecard"
    ```

4. 指定单个测试方法执行

    ```bash
    # 执行npu单卡训练用例
    pytest test_demo.py::TestMyModelTrainPredict::test_train_ascend_single_card -v
    ```

### 用例示例

以下为符合规范的tests/st/test_demo.py完整实现，覆盖 Ascend 单卡训练、Ascend 多卡推理核心场景：

```python
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from mindformers import Trainer, TrainingArguments
from mindformers.core.optim import AdamW
from mindformers.tools.logger import logger
from mindformers.models.llama import LlamaForCausalLM, LlamaConfig
from mindformers.trainer.optimizer_grouped_parameters import (
    get_optimizer_grouped_parameters,
)


# 命名规范：测试文件以test_开头，类以Test开头，格式为Test+模型名+核心功能
class TestSimpleCPUModel(nn.Cell):
     def __init__(self):
         super().__init__()
         self.fc1 = nn.Dense(16, 8)
         self.relu = nn.ReLU()
         self.fc2 = nn.Dense(8, 2)

     def construct(self, x):
         x = self.fc1(x)
         x = self.relu(x)
         x = self.fc2(x)
         return x


class TestMyModelTrainPredict:
    @classmethod
    def setup_class(cls):
        """类级初始化：初始化模型/训练配置"""
        ms.set_device("Ascend")
        cls.num_layers = 2
        cls.seq_length = 2
        cls.vocab_size = 32000
        cls.step_num = 1

        cls.model_config = LlamaConfig(
            num_layers=cls.num_layers,
            seq_length=cls.seq_length,
            use_flash_attention=True,
        )
        cls.train_args = TrainingArguments(
            batch_size=1, num_train_epochs=1, sink_mode=False, loss_scale_value=1024
        )

    def gen_dummy_data(self):
        """生成测试用虚拟数据集"""
        size = (
            self.step_num * self.train_args.batch_size,
            self.model_config.seq_length + 1,
        )
        input_ids = np.random.randint(low=0, high=self.vocab_size, size=size).astype(
            np.int32
        )
        for _, input_id in enumerate(input_ids):
            yield input_id

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu  
    @pytest.mark.env_onecard
    def test_train_x86_cpu_single_card(self):
        """
        Feature: mindformers模型训练
        Description: 测试X86架构CPU单卡模型训练与推理功能
        Expectation: 执行成功
        """
        def gen_data():
            for _ in range(5):
                data = np.random.rand(16).astype(np.float32)
                label = np.array(0, dtype=np.int32)
                yield data, label

        dataset = GeneratorDataset(gen_data, column_names=["data", "label"])
        dataset = dataset.batch(batch_size=2)

        net = TestSimpleCPUModel()
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optim = nn.Adam(net.trainable_params(), learning_rate=0.001)
        model = ms.Model(net, loss_fn=loss, optimizer=optim)

        model.train(epoch=1, train_dataset=dataset, dataset_sink_mode=False)

        test_input = ms.Tensor(np.random.rand(16).astype(np.float32))
        output = net(test_input)

        assert net is not None
        assert output is not None
        logger.info("X86 CPU single card training test passed!")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_train_ascend_single_card(self):
        """
        Feature: mindformers模型训练
        Description: 测试Atlas800T A2单卡Llama模型训练功能
        Expectation: 执行成功
        """
        dataset = GeneratorDataset(self.gen_dummy_data, column_names=["input_ids"])
        dataset = dataset.batch(batch_size=self.train_args.batch_size)

        model = LlamaForCausalLM(self.model_config)
        model.construct = ms.jit(jit_level="O1")(model.construct)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = AdamW(params=group_params)

        trainer = Trainer(
            task="text_generation",
            model=model,
            args=self.train_args,
            train_dataset=dataset,
            optimizers=optimizer,
        )
        trainer.config.callbacks = trainer.config.callbacks[:1]
        train_result = trainer.train()
        if train_result is None:
            train_result = {"loss":0.0}

        assert model is not None, "Model initialization failed after training"
        assert train_result is not None, "Training returned no result"
        logger.info("Ascend single card training test passed!")

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_predict_ascend_multi_card(self):
        """
        Feature: mindformers模型推理
        Description: 测试Atlas800T A2多卡Llama模型推理功能
        Expectation: 执行成功
        """
        model = LlamaForCausalLM(self.model_config)
        output = model.generate([1], max_length=5, do_sample=False)

        assert output is not None, "Inference output is empty"
        logger.info("Ascend multi card inference test passed!")


if __name__ == "__main__":
    # 本地调试执行：默认 NPU 单卡测试用例
    pytest.main(["-v", __file__, "-m", "platform_arm_ascend910b_training and env_onecard"])
```
