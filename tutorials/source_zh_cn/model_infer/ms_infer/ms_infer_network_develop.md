# 从零构建大语言模型推理网络

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/model_infer/ms_infer/ms_infer_network_develop.md)

## 大语言模型主干网络

当前主流的大语言模型主干网络都以基于transformer结构为主的，其中最为重要的就是Self-Attention机制的计算，以Qwen2大语言模型为例，下图简单描述了其主干网络结构：

![Qwen2网络结构](images/llm_qwen2_network_arch.png)

由此可见，Qwen2的核心层主要分为以下几部分：

- **Embedding**：将每个token对应的索引转换成一个向量，实现特征分散效果，类似onehot向量化，embedding的权重会参与训练过程，可以更好的适配语言模型中上下文语义，其实现就是一个Embedding算子既可完成。

- **DecodeLayer**：即Transformer结构，是大语言模型关键计算模块，通常根据配置不同，会重复多层计算，每一层实际就是一个Transformer结构。

- **RmsNorm&Linear**：输出线性归一层，在Transformer结构计算完后，将结果归一成和模型词表一样的维度，最终输出成每个token的概率分布返回。

使用MindSpore大语言模型推理构建网络，可以根据MindSpore提供的算子自己拼装，下面以Qwen2模型为例，简单描述如何构建模型过程。

## Qwen2ForCausalLM

Qwen2模型通常会对模型结构进行一定的封装成相关业务的模型，Qwen2ForCausalLM就是Qwen2面向语言处理和对话类业务的封装。

由于Qwen2大语言模型中的配置参数比较多，为了方便后续处理，我们先定义主要会用到的公共数据结构，主要包括模型配置（Qwen2Config）和模型输入（Qwen2ModelInput），下面是其对应的代码实现：

```python
import json
from typing import Optional, Type

from mindspore import Tensor

@dataclass
class Qwen2Config:
    """Qwen2 Config, the key-value is almost the same with config.json in HuggingFace"""
    architectures: Optional(List[str]) = None
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 3584
    initializer_range: float = 0.02
    intermediate_size: int = 18944
    max_position_embedding: int = 32768
    max_window_layers: int = 28
    model_type: str = "qwen2"
    num_attention_heads: int = 28
    num_hidden_layers: int = 28
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: Optional[int] = 131072
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformer_version: str = "4.41.2"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 152064
    param_dtype: Optional[Type] = dtype.bfloat16   # this is mindspore datatype as huggingface use str as dtype

    @classmethod
    def from_json(cls, json_path: str) -> Qwen2Config:
        with open(json_path) as f:
            data = json.load(f)
        config = cls(**data)
        return config

class Qwen2ModelInput:
    input_ids: Tensor
    positions: Tensor
    batch_valid_length: Tensor
    is_prefill: bool
    attn_mask: Tensor
    kv_cache: Optional[Tuple[Tensor, Tensor]] = None
    hidden_state: Optional[Tensor] = None
    residual: Optional[Tensor] = None
    block_tables: Tensor
    q_seq_lens: Tensor
```

其中，Qwen2Config配置和HuggingFace的配置基本一致，具体请参考Qwen2的官方文档，唯一的区别是此处用param_dtype替换了torch_dtype，由于mindspore的datatype类型与torch的不一致，因此我们这里直接使用单独的字段进行配置，此例子中，我们都会使用bfloat16类型；Qwen2ModelInput定义了模型的输入，包括主要的单词index，和KVCache等MindSpore推理优化特性所需要的数据。

接下来，我们通过Qwen2ForCausalLM类，将模型的主要接口定义清楚，下面是具体实现：

```python
from typing import Optional, Type

from mindspore import nn, load_checkpoint, load_param_into_net

class Qwen2ForCausalLM(nn.Cell):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = Qwen2Linear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            param_dtype=config.param_dtype,
            bias+False
        )

    def load_weight(self, weight_path: str):
        weight_dict = {}
        for path in glob(weight_path + "/*.safetensors")
            weight_dict.update(load_checkpoint(path, format="safetensors"))
        
        param_not_load, ckpt_not_load = load_param_into_net(self, weight_dict, strict_load=False)
        print(f"qwen2 load weight successful")

    def init_kv_cache(self, batch_size: int, max_seq_length: int):
        self.model.init_kv_cache(batch_size, max_seq_length)

    def construct(self, model_input: Qwen2ModelInput):
        hidden_states = self.model(model_input)
        logits = self.lm_head(hidden_states)[:, -1]
        return logits
```

由代码可见，Qwen2ForCausalLM主要有3个核心接口：

- **load_weight**：从HuggingFace官网模型加载权重，并且按照网络脚本注入到模型中。

- **init_kv_cache**：初始化KVCache结构，以使能全量和增量推理能力。

- **construct**：主要推理计算，会调用子模块一层层完成计算。

由construct可以看出，模型核心分为主干网络计算和最后一个lm_head的linear计算，将hidden_size的特征转换成vocab_size的词表概率分布。

## Qwen2Model

Qwen2Model是qwen2模型的主要网络，其组成主要分为两部分：一是将输入转换成特征的embedding层，另一个是n层Transformer的decoder结构。

### Embedding

embedding层逻辑比较简单，就是根据输入单词id，获取对应的hidden_size的特征数据（此数据也是训练权重的一部分），通过一个gather算子就可以实现，代码如下：

```python
from typing import Optional, Type

from mindspore import nn, ops, mint, Parameter, Tensor

class VocabEmbedding(nn.Cell):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        
        self。num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size

        self.gather = ops.Gather

        self.weight = Parameter(
            mint.zeros(
                (self.num_embeddings, self.embedding_dim),
                dtype=config.param_dtype
            ),
            requires_grad=False
        )

    def construct(self, input_ids: Tensor):
        return self.gather(input_ids)
```

### DecoderLayer

DecoderLayer是transformer网络的核心计算单元，其主要计算都包含在这一层中，从qwen2的网络结构图可以看出，主药包含Attention、MLP、Linear、RmsNorm、Rope等网络层，为了方便开发，我们先完成这些网络层的构建。

#### RmsNorm

RmsNorm是当前大语言模型中常用的归一算法，在MindSpore中有直接可以使用的算子，我们只需要对应的实现权重创建即可。同时，由于RmsNorm经常会有残差计算，因此我们实现了残差融合计算在网络层中，代码如下：

```python
from typing import Optional, Type

from mindspore import nn, ops, mint, Parameter, Tensor

class RmsNorm(nn.Cell):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.rms_norm = ops.RmsNorm(config.rms_norm_eps)

        self.weight = Parameter(
            mint.ones(
                config.hidden_size,
                dtype=config.param_dtype
            ),
            requires_grad=False
        )

    def construct(self, x: Tensor, residual: Tensor):
        if residual is not None:
            x = x + residual
            residual = x
        output = self.rms_norm(x, self.weight)[0]
        if residual is None:
            return output, None
        return output, residual
```

#### Linear

Linear层实际就是一个线性变换，其主要的计算逻辑就是一个矩阵乘法MatMul，不过会根据具体使用场景来判断是否要进行bias加法的偏差纠正（query、key、value转换时需要bias），我们将这些计算融入到一个网络结构中，代码如下：

```python
from typing import Optional, Type

from mindspore import nn, ops, mint, Parameter, Tensor

class Qwen2Linear(nn.Cell):
    def __init__(self, input_size: int, output_size: int, param_dtype: Optional[Type], enable_bias: bool) -> None:
        super().__init__()

        self.param_dtype = param_dtype
        self.input_size = input_size
        self.output_size = output_size
        self.enable_bias = enable_bias

        self.matmul = ops.MatMul(transpose_b=True)
        self.weight = Parameter(
            mint.zeros(
                (self.output_size, self.input_size)
                dtype=config.param_dtype
            ),
            requires_grad=False
        )

        if self.enable_bias:
            self.bias_add = ops.Add()
            self.bias = Parameter(
                mint.zeros(self.output_size, dtype=self.param_dtype)
            )

    def construct(self, input: Tensor):
        origin_shape = input.shape
        x = self.matmul(input.view(-1, origin_shape[-1]), self.weight)
        if self.enable_bias:
            x = self.bias_add(x, self.bias)
        return x.view(*origin_shape[:-1], -1)
```

其中，由于我们需要支持多batch计算，因此传入的input的shape可能是input_size的n倍，为了保证计算正确，我们保存了原始输入shape，并在计算完成后，重新通过view还原shape。

#### Rope

Rope算子是旋转位置编码，是为了能够让Attention能够更好的识别单词间距离的影响，会在query和key的特征上加上一个位置编码信息，rope算子由于其特性，可以采用一开始就计算好的结果，在使用时直接查表实现，因此可以通过gather和rope算子实现，具体计算可以参考旋转位置编码相关材料。

```python
import numpy as np
from typing import Optional, Type

from mindspore import nn, ops, mint, Parameter, Tensor

class Qwen2RotaryEmbedding(nn.Cell):
    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: int, dtype: Optional[Type]) -> None:
        super().__init__()

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        # format 2 is neox style
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.gather = ops.Gather()

        self.freqs_cos, self.freqs_sin = 

    def _compute_inv_freq(self) -> Tensor:
        freqs_base = mint.arange(0, self.rotary_dim, 2).astype(np.float32)
        freqs = 1.0 / (self.base ** (freqs_base / self.rotary_dim))
        return freqs

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq()
        t = np.arange(0, self.max_position_embeddings, 1).astype(np.float32)
        freqs = np.outer(t, freqs)
        emb = np.concatenate((freqs, freqs), axis=1)
        freqs_cos = np.cos(emb)
        freqs_sin = np.sin(emb)

        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(self, positions: Tensor, query: Tensor, key: Tensor, batch_valid_length: Tensor):
        bs, seq_len, _. _ = query.shape
        query = query.view(bs * seq_len, -1)
        key = key.view(bs * seq_len, -1)

        query = query.contiguous()
        key = key.contiguous()

        if is_prefill:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            freqs_cos = self.gather(self.freqs_cos, positions.view(-1), 0)
            freqs_sin = self.gather(self.freqs_sin, positions.view(-1), 0)
        
        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin, batch_valid_length)
```

#### Attention

Attention层是由多个Linear，Rope等组成的，其中Attention分数计算MindSpore提供了FlashAttention和PagedAttention两个融合大算子来提升推理性能，根据网络结构，可以构造如下网络代码：

```python
import numpy as np
from typing import Optional, Type

from mindspore import nn, ops, mint, Parameter, Tensor

class Qwen2MLP(nn.Cell):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

        self.up_proj = Qwen2Linaer(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            param_dtype=config.param_dtype,
            bias=False
        )
        self.gate_proj = Qwen2Liner(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            param_dypte=config.param_dtype,
            bias=False
        )
        self.down_proj = Qwen2Linear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            param_dtype=config.param_dtype,
            bias=False
        )
        self.act_fn = ops.silu

    def construct(self, x: Tensor) -> Tensor:
        output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return output
```


