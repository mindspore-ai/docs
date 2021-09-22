# Copyright 2020 Huawei Technologies Co., Ltd
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

'''
Functional Cells used in Bert finetune and evaluation.
'''
import numpy as np
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal, initializer
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import Parameter, ParameterTuple
from mindspore import dtype as mstype
from mindspore.nn import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication import get_group_size
from mindspore import context
from .bert_model import BertModel
from .bert_for_pre_training import clip_grad

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)

_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class BertPoetryCell(nn.TrainOneStepWithLossScaleCell):
    """
    Specifically defined for finetuning where only four inputs tensor are needed.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):

        super(BertPoetryCell, self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(
            get_by_list=True,
            sens_param=True)
        self.reducer_flag = False
        self.allreduce = ops.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("mirror_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = ops.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = ops.FloatStatus()
            self.addn = ops.AddN()
            self.reshape = ops.Reshape()
        else:
            self.alloc_status = ops.NPUAllocFloatStatus()
            self.get_status = ops.NPUGetFloatStatus()
            self.clear_before_grad = ops.NPUClearFloatStatus()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = ops.LessEqual()
        self.hyper_map = ops.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")

    def construct(self,
                  input_ids,
                  token_type_id,
                  pad_mask,
                  sens=None):
        """construct BertPoetryCell"""

        weights = self.weights
        loss = self.network(input_ids,
                            token_type_id,
                            pad_mask)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        grads = self.grad(self.network, weights)(input_ids,
                                                 token_type_id,
                                                 pad_mask,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(ops.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond)
        return ops.depend(ret, succ)



class BertPoetryModel(nn.Cell):
    """BertPoetryModel"""
    def __init__(self, config, is_training, num_tokens, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertPoetryModel, self).__init__()
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.num_tokens = num_tokens
        idx = np.arange(config.seq_length)
        mask = idx[None, :] <= idx[:, None]
        self.mask = Tensor([mask], mstype.float32)
        self.MLM_Dense = nn.Dense(config.hidden_size, config.hidden_size,\
                                has_bias=True, weight_init=TruncatedNormal(0.02),\
                                activation='gelu').to_float(mstype.float16)
        self.layer_norm = nn.LayerNorm((config.hidden_size,))
        self.matmul = ops.MatMul(transpose_b=True)
        self.biasadd = Parameter(initializer('zero', self.num_tokens), name='MLM_output_biasadd')
        self.softmax = ops.Softmax(axis=-1)
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.batch_matmul = ops.BatchMatMul()
        ones = np.ones(shape=(config.batch_size, config.seq_length, config.seq_length))
        self.lower_triangle_mask = Tensor(np.tril(ones), dtype=mstype.float32)
        self.multiply = ops.Mul()

    def construct(self, input_ids, token_type_id, input_mask):
        """construct BertPoetryModel"""
        input_shape = ops.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        input_mask = self.cast(input_mask, mstype.float32)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.batch_matmul(mask_left, mask_right)
        attention_mask = self.multiply(attention_mask, self.lower_triangle_mask)


        sequence_output, _, embedding_tables = self.bert(input_ids, token_type_id, attention_mask)
        bert_output = ops.Reshape()(sequence_output, (-1, self.hidden_size))
        MLM_output = self.MLM_Dense(bert_output)
        MLM_output = self.layer_norm(MLM_output)
        embedding_tables = ops.Cast()(embedding_tables, mstype.float16)
        output = self.matmul(MLM_output, embedding_tables)
        output = ops.Cast()(output, mstype.float32)
        output = output + self.biasadd
        output = ops.Reshape()(output, (-1, self.seq_length, self.num_tokens))

        logits = self.softmax(output)
        return logits




class BertPoetry(nn.Cell):
    """BertPoetry"""
    def __init__(self, model, config, is_training, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertPoetry, self).__init__(auto_prefix=False)
        self.num_tokens = 3191
        self.poetry = model
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.reshape = ops.Reshape()
        self.neg = ops.Neg()
        self.cast = ops.Cast()
        self.last_idx = (-1,)
        self.log = ops.Log()
        self.max = ops.ArgMaxWithValue(axis=-1)

    def construct(self, input_ids, token_type_id, pad_mask):
        """construct BertPoetry"""
        logits = self.poetry(input_ids, token_type_id, pad_mask)
        logits = logits[:, :127, :]
        label_ids = input_ids[:, 1:]

        one_hot_labels = self.onehot(label_ids, self.num_tokens, self.on_value, self.off_value)
        per_example_loss = self.neg(self.reduce_sum(one_hot_labels * self.log(logits), self.last_idx))
        loss = per_example_loss * pad_mask[:, 1:]
        loss = self.reduce_sum(loss) / self.reduce_sum(pad_mask)
        return_value = self.cast(loss, mstype.float32)
        return return_value



class BertLearningRate(nn.WarmUpLR):
    """
    Warmup-decay learning rate for Bert network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__(learning_rate, warmup_steps)
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = nn.WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = nn.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.greater = ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()

    def construct(self, global_step):
        """construct BertLearningRate"""
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr
