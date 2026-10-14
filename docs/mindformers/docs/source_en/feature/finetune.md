# Full-Parameter Fine-Tuning and LoRA Fine-Tuning

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.11.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.11.0/docs/mindformers/docs/source_en/feature/finetune.md)

MindSpore Transformers' dynamic graph (PyNative) mode supports **full-parameter fine-tuning** and **LoRA efficient fine-tuning**. Full-parameter fine-tuning updates all model parameters; LoRA fine-tuning freezes the base model parameters and only trains low-rank adapters injected into regular `Linear` layers and MoE routed expert `GroupedMLP` layers.

Both fine-tuning methods are launched via `run_mindformer.py --mode 1` and share the `PynativeTrainer` with pre-training. Before starting, prepare:

- Complete Safetensors base weights matching the model architecture;
- The model's complete dynamic graph YAML — you cannot use only the incremental snippets shown in this document;
- A fine-tuning dataset and a tokenizer matching the base model;
- MindSpore Transformers and HyperParallel installed according to the [Installation Guide](../installation.md).

For dataset preparation and configuration, refer to the [HuggingFace Dataset section in the Dataset documentation](./dataset.md#hugging-face-datasets).

## 1. Full-Parameter Fine-Tuning

Based on the YAML file provided by MindSpore Transformers, modify the `checkpoint` section:

```yaml
checkpoint:
  enable_save: true
  save_path: "./output/full_finetune"
  save_interleaved_steps: 500
  load_path: "/path/to/base_model"
  no_load_optim: true
```

| Field | Description |
|---|---|
| `load_path` | Loads the complete base weights. Can be a HuggingFace model directory, a training save root directory, or an `iteration_*` directory |
| `no_load_optim` | Set to `true` for the first fine-tuning run to load only model weights and create a new optimizer; set to `false` for resumable training |

The remaining `training`, `optimizer`, `lr_scheduler`, `model`, `parallelism`, and data configurations follow the target model's dynamic graph training YAML. For the launch command, see [Starting Training](#4-starting-training).

## 2. LoRA Fine-Tuning

LoRA injects low-rank matrices alongside the frozen base weights. The forward computation of a regular linear layer is:

$$y = x W^{T} + \frac{\alpha}{r}(x A^{T})B^{T}$$

where $W$ is frozen and only $A/B$ participate in training. The default initialization is $A \sim \mathcal{N}(0,\sigma^2)$, $B=0$, so at the start of training $\Delta W=0$ and the model output matches the base model.

### 2.1 Configuring Base Loading and Adapter Saving

It is recommended to save only the LoRA adapters to reduce checkpoint size:

```yaml
checkpoint:
  enable_save: true
  save_path: "./output/lora_finetune"
  save_interleaved_steps: 500

  # First LoRA fine-tuning: load complete base weights from this directory
  base_load_path: "/path/to/base_model"
  load_path: ""

  # Save only LoRA parameters; base_load_path must be set when loading
  save_trainable_only: true

  # Preserve optimizer state for resumable training
  no_save_optim: false
  no_load_optim: false
```

When `save_trainable_only: true`, only adapter parameters are saved, along with a `mindformers_adapter_config.json` file recording the LoRA configuration, base path, and base fingerprint. When loading or resuming training, both `base_load_path` and `load_path` must be configured.

When `save_trainable_only: false`, both the complete base parameters and adapter parameters are saved, and the checkpoint size is close to that of full-parameter saving.

### 2.2 Configuring LoRA Injection Targets

Add `lora_config` at the top level of the YAML:

```yaml
lora_config:
  target_modules: ".*linear_qkv.*|.*linear_proj.*|.*linear_fc1.*|.*linear_fc2.*"
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.0
```

| Field | Default | Description |
|---|---|---|
| `target_modules` | `""` | Required regular expression. `Linear` or `GroupedMLP` modules whose dotted module path matches the regex will be replaced with LoRA layers; an error is raised if no supported layer is matched |
| `exclude_layers` | `None` | Optional regular expression; excludes modules already matched by `target_modules` |
| `lora_rank` | `8` | Adapter rank $r$ |
| `lora_alpha` | `16` | Scaling numerator; effective scaling is $\alpha/r$ |
| `lora_dropout` | `0.0` | Dropout probability for the adapter branch input |
| `lora_a_std` | `0.01` | Standard deviation for normal initialization of `lora_a` |
| `lora_b_std` | `0.0` | Standard deviation for normal initialization of `lora_b`; keep at `0.0` for standard fine-tuning |

Common target names in regular dynamic graph Transformers include `linear_qkv`, `linear_proj`, `linear_fc1`, and `linear_fc2`. To inject only into attention layers:

```yaml
lora_config:
  target_modules: ".*linear_qkv.*|.*linear_proj.*"
```

MoE routed experts are implemented by `GroupedMLP`, and module names typically end with `experts`. To inject LoRA only into routed experts:

```yaml
lora_config:
  target_modules: '.*\.experts$'
```

To match both regular linear layers and routed experts, combine the regexes with `|`. For example, to inject LoRA into both attention layers and MoE routed experts:

```yaml
lora_config:
  target_modules: '.*linear_qkv.*|.*linear_proj.*|.*\.experts$'
```

> Module naming may differ across models; refer to the target model code and the module paths in the startup logs.

### 2.3 LoRA Resumable Training

When resuming training with an adapter-only checkpoint, specify both the original base directory and the adapter save directory:

```yaml
checkpoint:
  base_load_path: "/path/to/base_model"
  load_path: "./output/lora_finetune"
  save_path: "./output/lora_finetune"
  save_trainable_only: true
  no_save_optim: false
  no_load_optim: false
```

The framework first verifies the base fingerprint recorded in `mindformers_adapter_config.json`, then loads the base, adapters, and optimizer state in order.

When resuming training with a full checkpoint, `load_path` points to the full save directory and `base_load_path` is not required:

```yaml
checkpoint:
  load_path: "./output/lora_finetune"
  save_path: "./output/lora_finetune"
  save_trainable_only: false
  no_save_optim: false
  no_load_optim: false
```

The framework restores the base parameters, LoRA parameters, optimizer state, and training step from `load_path`. The `lora_config` used when resuming must match the one used when the checkpoint was saved.

## 3. Merging LoRA Weights

LoRA weight merging is used to merge adapter increments into the base parameters to produce complete weights that do not depend on LoRA adapters, for inference or deployment.

For adapter-only save mode, pass the base and adapter directories separately:

```bash
python mindformers/tools/transform_safetensors_lora.py \
  --base_ckpt_path /path/to/base_model \
  --adapter_ckpt_path ./output/lora_finetune \
  --dst_ckpt_path ./output/merged.safetensors
```

| Parameter | Description |
|---|---|
| `--base_ckpt_path` | Path to the complete base weights. Supports a single Safetensors file, a single-card directory, a training save root directory with a tracker, and a distributed directory containing `metadata.json` |
| `--adapter_ckpt_path` | Optional standalone LoRA checkpoint. Required for adapter-only save mode; the tool verifies its base fingerprint |
| `--dst_ckpt_path` | Merged output file; the `.safetensors` suffix can be omitted, and the output is always a single complete weight file |
| `--lora_alpha` | Optional. By default read from `mindformers_adapter_config.json` or `adapter_config.json`; falls back to `16` if unreadable. If passed explicitly, it must match the training configuration |

If the checkpoint contains both base and LoRA parameters, `--adapter_ckpt_path` can be omitted:

```bash
python mindformers/tools/transform_safetensors_lora.py \
  --src_ckpt_path ./output/lora_full_checkpoint \
  --dst_ckpt_path ./output/merged.safetensors \
  --lora_alpha 16
```

`--src_ckpt_path` is a compatible alias for `--base_ckpt_path`. The merge tool supports regular linear layer LoRA and MoE `GroupedMLP` expert LoRA, and automatically ignores optimizer shards.

## 4. Starting Training

Run from the root of the MindSpore Transformers repository:

```bash
# Single card
python run_mindformer.py --config /path/to/finetune.yaml --mode 1

# Single machine, 8 cards
bash scripts/msrun_launcher.sh \
  "run_mindformer.py --config /path/to/finetune.yaml --mode 1" 8
```

For multi-card, `dp_replicate * dp_shard * cp * tp * pp` must equal the number of processes; `global_batch_size`, `local_batch_size`, and the parallelism dimensions must also satisfy the constraints in [Distributed Parallel Training](./parallel_training.md). The appearance of `Running MindFormers in PYNATIVE_MODE.` in the startup logs indicates that dynamic graph training has been entered.

## 5. Comparison of Full-Parameter Fine-Tuning and LoRA Fine-Tuning

| Dimension | Full-Parameter Fine-Tuning | LoRA Fine-Tuning |
|---|---|---|
| Trainable parameters | All model parameters | Only LoRA adapters, base frozen |
| Optimizer state and memory | High | Lower |
| Base loading | `checkpoint.load_path` | `checkpoint.base_load_path` recommended |
| Save artifacts | Complete model checkpoint | Recommended to save only adapters and their manifest |
| Pre-deployment processing | Complete weights matching the configuration can be used directly | Adapter-only mode requires merging with the base first |

## Related Documents

- [Starting Tasks](./start_task.md)
- [Configuration File Description](./configuration.md)
- [Weight Saving and Loading](./save_load_checkpoint.md)
- [Resumable Training](./resume_training.md)
- [Dataset](./dataset.md)
- [Distributed Parallel Training](./parallel_training.md)
- [Hyperparameters and Optimizers for Training](./training_hyperparameters.md)
