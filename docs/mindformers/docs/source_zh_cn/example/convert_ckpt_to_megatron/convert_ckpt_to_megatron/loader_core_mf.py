# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
This module provides a loader for converting MindSpore safetensors to Megatron format.

Only supports converting a single complete (unsharded) MindSpore checkpoint into Megatron's distributed checkpoint
format (i.e., generates a single mp_rank_00 file).
Does not support direct conversion of sharded or multi-rank checkpoints.

This file should be copied to the Megatron-LM repository under tools/checkpoint/ and used together with other scripts.
Only supports models with SelfAttention + MLP. MLA and MoE (MoEt) models are not supported.

Args:
    --true-vocab-size: (int, optional) Original size of vocab; if specified, trims padding from embedding table.
    --vocab-file: (str, optional) Path to a vocab file. If specified, determines vocab size to trim padding.
    --megatron-path: (str, optional) Base directory of Megatron repository.
    --position-embedding-type: (str) Type of position embedding. Choices: ['learned_absolute', 'rope'].
    --loader-transformer-impl: (str) Which Transformer implementation to use. Choices: ['local', 'transformer_engine'].
    --num-layers: (int) Number of transformer layers.
    --seq-length: (int) Sequence length.
    --padded-vocab-size: (int) Padded vocabulary size.
    --hidden-size: (int) Hidden size.
    --ffn-hidden-size: (int) FFN hidden size.
    --num-attention-heads: (int) Number of attention heads.
    --num-query-groups: (int, optional) Number of query groups.
    --normalization: (str) Normalization type.
    --max-position-embeddings: (int) Maximum position embeddings.
    --add-bias-linear: (bool) Whether to add bias in linear layers.
    --swiglu: (bool) Whether to use swiglu activation.
    --tokenizer-type: (str) Tokenizer type.
    --ms2torch-ckpt-path: (str) Output path for the converted Megatron checkpoint.

"""
import glob
import os
import argparse
from safetensors.torch import load_file
import torch
from loader_core import MegatronCheckpointLoaderLLM

MS2TORCH_CKPT_PATH = "./ms2pt_checkpoint"


def add_arguments(parser):
    """Add command-line arguments relevant to Megatron model loading."""
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Original size of vocab; if specified, trims padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to a vocab file. If specified, determines vocab size to trim padding.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Type of position embedding.')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')

    group.add_argument('--num-layers', type=int, default=512)
    group.add_argument('--seq-length', type=int, default=2048)
    group.add_argument('--padded-vocab-size', type=int, default=128000)
    group.add_argument('--hidden-size', type=int, default=512)
    group.add_argument('--ffn-hidden-size', type=int, default=128)
    group.add_argument('--num-attention-heads', type=int, default=64)
    group.add_argument('--num-query-groups', type=int, default=None)
    group.add_argument('--normalization', default="RMSNorm")
    group.add_argument('--max-position-embeddings', type=int, default=2048)
    group.add_argument('--add-bias-linear', action='store_true', default=False,
                       help='Add bias in linear layers (flag, set True if specified).')
    group.add_argument('--swiglu', action='store_true', default=False,
                       help='Use swiglu activation (flag, set True if specified).')
    group.add_argument('--tokenizer-type', default="HuggingFaceTokenizer")

    group.add_argument('--ms2torch-ckpt-path', default=MS2TORCH_CKPT_PATH)


class MegatronCheckpointLoaderLLMFromMS(MegatronCheckpointLoaderLLM):
    """Loader for converting MindSpore safetensors to Megatron distributed checkpoint format."""

    def convert_ms_ckpt_to_pt(self):
        """Convert MindSpore checkpoint to Megatron PyTorch checkpoint."""
        tensors = {}

        if os.path.isdir(self.args.load_dir):
            safetensor_files = sorted(glob.glob(os.path.join(self.args.load_dir, "*.safetensors")))
            if not safetensor_files:
                raise FileNotFoundError(f"No .safetensors files found in {self.args.load_dir}")
            for file in safetensor_files:
                tensors.update(load_file(file))
        else:
            tensors = load_file(self.args.load_dir)

        new_tensors = {}
        for k, v in tensors.items():
            if "dropout" in k:
                continue
            new_tensors[k] = v
        new_tensors["decoder.final_layernorm._extra_state"] = None

        state_dict = {"model": new_tensors}

        args = argparse.Namespace(
            num_layers=self.args.num_layers,
            seq_length=self.args.seq_length,
            padded_vocab_size=self.args.padded_vocab_size,
            hidden_size=self.args.hidden_size,
            ffn_hidden_size=self.args.ffn_hidden_size,
            num_attention_heads=self.args.num_attention_heads,
            num_query_groups=self.args.num_query_groups,
            normalization=self.args.normalization,
            max_position_embeddings=self.args.max_position_embeddings,
            position_embedding_type=self.args.position_embedding_type,
            add_bias_linear=self.args.add_bias_linear,
            swiglu=self.args.swiglu,
            fp16=False,
            bf16=True,
            tokenizer_type=self.args.tokenizer_type,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            sequence_parallel=False,
            apply_query_key_layer_scaling=False,
            num_experts=None,
        )
        state_dict['args'] = args
        state_dict["iteration"] = 1
        state_dict["checkpoint_version"] = 4.0

        os.makedirs(os.path.join(self.args.ms2torch_ckpt_path, "iter_0000001/mp_rank_00/"), exist_ok=True)
        torch.save(state_dict, os.path.join(self.args.ms2torch_ckpt_path, "iter_0000001/mp_rank_00/model_optim_rng.pt"))

        with open(os.path.join(self.args.ms2torch_ckpt_path, 'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')

        self.args.load_dir = self.args.ms2torch_ckpt_path

    def load(self):
        """Convert and load the checkpoint using the parent loader."""
        self.convert_ms_ckpt_to_pt()
        super().load()


def load_checkpoint(queue, args):
    """
    Required top-level function that creates the loader,
    calls its .load(), and handles exceptions by signaling 'exit'.
    """
    loader = MegatronCheckpointLoaderLLMFromMS(args, queue)
    try:
        loader.load()
    except Exception as e:
        queue.put("exit")
        raise e
