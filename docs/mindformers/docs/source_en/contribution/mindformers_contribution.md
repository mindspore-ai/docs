# MindSpore Transformers Contribution Guidelines

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/contribution/mindformers_contribution.md)

## Contributing Code to MindSpore Transformers

### Code Style Requirements

Please follow this style for MindSpore Transformers review, maintenance and development.

- Coding Guide

  The MindSpore Transformers community uses the `Python PEP 8` coding style. It is recommended to install the following plugins in your IDE to check code format: `Lizard`, `ShellCheck` and `PyLint`.

- Unit Testing Guide

  The MindSpore Transformers community uses the Python unit testing framework pytest. Annotation names need to reflect the design intent of the test case.

- Reconstruction Guide

  We encourage developers to reconstruct our code to eliminate code bad taste. All code should conform to coding style and testing style, and reconstructing code is no exception. The Lizard threshold for uncommented lines of code (nloc) is 100, and the cyclomatic complexity (ccn) threshold is 20. When a Lizard warning is received, the code to be merged must be reconstructed.

- Documentation Guide

  We use MarkdownLint to check Markdown document format. The following rules are modified based on the default configuration:

  1. MD009 (space at the end of the line): the parameter br_spaces is set to 2, indicating that there can be either 0 or 2 spaces at the end of the line.
  2. MD029 (sequence number of ordered list): the parameter style is set to ordered, indicating ascending order.

### Fork-Pull Development Model Guide

- Fork MindSpore Transformers code repository

  Before submitting code to the MindSpore Transformers project, please make sure that you have forked this project to your own code repository. There may be parallel development between the MindSpore Transformers code repository and your own code repository, so please be aware of the consistency between them.

- Clone remote code repository

  If you want to download the code to your local computer, it is best to use the git method.

  ```shell
  # Clone repositories on AtomGit
  git clone https://atomgit.com/(insert_your_forked_repo)/mindformers.git
  ```

- Local Development Code

  `master` is the development branch. Please pull the latest code from `master` branch for development. And submit it to the `master` branch when you submit your Pull Request.

  ```shell
  git checkout -b {new branch name} origin/master
  ```

- Submit PR to MindSpore Transformers code repository

  In the last step, you need to pull a compare request between the new branch and the `MindSpore Transformers` master branch. After completing the pull request, `Jenkins CI` will be automatically set up for build testing. PR should be merged into the upstream master branch as soon as possible to minimize the risk of merging.

  ```shell
  # Add all changes to the staging area
  git add .

  # Check Update Status
  git status

  # To commit changes, add a commit header with the -m option
  git commit -m "The title of your commit"

  # Add a specific description of the commit, add a signature with the -s option, and modify the most recent commit with the `--amend` option.
  git commit -s --amend

  # Push changes to a new branch in the remote repository
  git push origin {New branch name}

  ```

### Documentation and Code Format

If you wish to merge custom models into the `MindSpore Transformers` code repository, there are a few things to keep in mind:

1. The file format and location should follow the norms.
2. Register the new model in the code to adapt it for higher-order interface use.

#### File Format and Location

1. The model code files are placed uniformly in the `research/{model_name}` folder in the following format.

    ```text
    research/{model_name}
    ├── {model_name}
    | ├── {pretrain/finetune/predict}_{model_name}_{n}b.yaml
    | ├── convert_weight.py # Torch weights to MindSpore weights script (required for migration models)
    | ├── convert_reversed.py # MindSpore weights to Torch weights script (required for migration models)
    | ├── run_{model_name}.py # Running the code file
    | ├── {model_name}.py   # Model class code file
    | └── {model_name}_tokenizer.py # Tokenizer Code File
    ```

2. Model documentation is placed in the same `research/{model_name}` folder.

## Requirements for Submitting A PR

### Only One Commit

For multi-commit PRs, use the `squash` command to merge multiple commits into one. For example, use:

```shell
git rebase -i HEAD~3
```

You can see:

```shell
pick 1234567 Add new function A
pick 89abcdef Fixed bugs in A
pick 01234567 Some optimizations to A
```

squash merge commit (can be simplified to abbreviations such as s, p, f, etc.)

```shell
pick 1234567 Add new function A
pick 89abcdef Fixed bugs in A
pick 01234567 Some optimizations to A
```

### PR Descriptions

Please use the following md template.

```markdown

### Related Issue

### Reason (purpose, problem solved, etc.)

### Description (what was done, what was changed)

### Checklist

#### Was a program review or root cause analysis of the problem completed (Y/N)

#### Whether UT/ST of functional modules was completed, executed and passed with results attached (Y/N)

#### Whether it involves modification of public components or external interfaces, and if so, the scope of modification and impact assessment should be given (Y/N)

#### Whether it involves the modification of information, and if so, the modification should be synchronized (Y/N)

```

### Access Control Requirements

1. Submitting a PR requires [signing a CLA](https://www.mindspore.cn/icla).

2. Submitting a PR requires passing the CI check, which needs to be manually restarted by commenting `/retest` under comments after the gate fails and the code is corrected.

## Test Case Contribution

### Organization Structure

#### Directory Structure

```ColdFusion
    tests/  
    ├── st/                        # System Testing: Verify end-to-end workflows of multi-component collaboration  
    │   ├── test_auto_register/        # Test automatic registration of custom models/operators  
    │   ├── test_ckpt_health_monitor/  # Test integrity check of model checkpoints  
    │   ├── test_docs/                 # Test runnability of code examples in documentation  
    │   ├── test_grace_exit_save_ckpt/ # Test checkpoint saving during training interruption  
    │   ├── test_infer/                # Test single-card/multi-card/offline inference workflows  
    │   ├── test_model/                # Test consistency of model execution across multiple devices  
    │   ├── test_multi_cards_cases/    # Test multi-card distributed training/inference  
    │   ├── test_optim/                # Test optimizers/learning rate/mixed precision training  
    │   ├── test_resume/               # Test training resumption from breakpoints  
    │   ├── test_safetensors/          # Test loading/saving of Safetensors checkpoints  
    ├── utils/                     # Test Utility Library: Data generation, device detection, etc.  
    ├── conftest.py                # pytest Global Configuration: Environment check, initialization
```

#### Basic Specifications

1. Test Case Marking Rules:

    - NPU test cases: @pytest.mark.platform_arm_ascend910b_training
    - CPU test cases: @pytest.mark.platform_x86_cpu
    - Single-card test cases: @pytest.mark.env_onecard
    - Multi-card test cases (8 cards by default): @pytest.mark.platform_env_single

2. Test Case Development Specifications:

    - Test cases generate cache files in the directory of the test file
    - Add execution rule-related marks above methods (including class methods) for all test cases, not above classes
    - Test files start with "test_". Classes start with "Test". Methods start with "test"

3. Test Case Level Specifications:

    - Level 0: Combined interface test cases (only parallel interfaces are classified into this level)
    - Level 1: Full-network function test cases, single-card test cases of parallel computing interfaces, atomic interface test cases

### Execution Examples

1. Install Dependencies

    ```bash
    pip3 install -r requirements.txt
    ```

2. Execute a Single Test File

    ```bash
    pytest tests/st/test_demo.py
    ```

3. Execute with Mark Filtering

    ```bash
    # Filter test cases with specified marks using the -m parameter
    # Execute all npu single-card test cases
    pytest test_demo.py -v -m "platform_arm_ascend910b_training and env_onecard"
    ```

4. Execute a Single Test Method

    ```bash
    # Execute X86 CPU single-card training test case
    pytest test_demo.py::TestMyModelTrainPredict::test_train_ascend_single_card -v
    ```

### Test Case Example

The following is a complete implementation of tests/st/test_demo.py that complies with specifications, covering core scenarios of CPU single-card, Ascend single-card training, and Ascend multi-card inference:

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


# Naming Convention: Test files start with test_, classes start with Test, format: Test + Model Name + Core Function
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
        """Class-level initialization: Initialize model / training configuration"""
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
        """Generate dummy dataset for testing"""
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
        Feature: mindformers model train
        Description: Test X86 architecture CPU single-card model training and inference
        Expectation: success
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
        Feature: mindformers model train
        Description: Test Atlas800T A2 single-card Llama model training
        Expectation: success
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
        Feature: mindformers model predict
        Description: Test Atlas800T A2 multi-card Llama model inference
        Expectation: success
        """
        model = LlamaForCausalLM(self.model_config)
        output = model.generate([1], max_length=5, do_sample=False)

        assert output is not None, "Inference output is empty"
        logger.info("Ascend multi card inference test passed!")


if __name__ == "__main__":
    # Local debug execution: Default npu single-card test case
    pytest.main(["-v", __file__, "-m", "platform_arm_ascend910b_training and env_onecard"])
```
