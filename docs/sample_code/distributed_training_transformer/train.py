# Copyright 2021 Huawei Technologies Co., Ltd

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
Train file for training transformers
"""
import argparse
from mindspore.parallel.nn import TransformerOpParallelConfig
from mindspore import Model
import mindspore.communication as D
from mindspore.context import ParallelMode
from mindspore.nn import PipelineCell
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.nn import AdamWeightDecay
from mindspore import context
from dataset import ToyDataset, Tokenzier
from model import Net


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def main():
    # Run the total forward model
    parser = argparse.ArgumentParser(description="PanguAlpha training")
    parser.add_argument('--train',
                        type=int,
                        default=1,
                        help="Running training or prediction.")
    parser.add_argument("--device_num",
                        type=int,
                        default=128,
                        help="Use device nums, default is 128.")
    parser.add_argument("--file_path",
                        type=str,
                        default="./output/wmt14.fr_en.txt",
                        help="Use device nums, default is 128.")
    parser.add_argument("--distribute",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Run distribute, default is true.")
    parser.add_argument("--micro_batch_num",
                        type=int,
                        default=1,
                        help="The micro batch num.")
    parser.add_argument('--src_len',
                        required=False,
                        type=int,
                        default=10,
                        help='The source sequence length.')
    parser.add_argument('--tgt_len',
                        required=False,
                        type=int,
                        default=10,
                        help='The target sequence length.')
    parser.add_argument('--vocab_size',
                        required=False,
                        type=int,
                        default=20000,
                        help='The vocab size.')
    parser.add_argument('--d_model',
                        required=False,
                        type=int,
                        default=128,
                        help='The hidden size of the model.')
    parser.add_argument('--encoder_layer',
                        required=False,
                        type=int,
                        default=1,
                        help='The number of the encoder layer.')
    parser.add_argument('--decoder_layer',
                        required=False,
                        type=int,
                        default=1,
                        help='The number of the decoder layer.')
    parser.add_argument('--pipeline_stage',
                        required=False,
                        type=int,
                        default=1,
                        help='The pipeline stage number.')
    parser.add_argument('--batch_size',
                        required=False,
                        type=int,
                        default=32,
                        help='The batch size of the inputs.')
    parser.add_argument('--lr',
                        required=False,
                        type=float,
                        default=0.0001,
                        help='The learnign rate of the training process.')
    parser.add_argument('--dp',
                        required=False,
                        type=int,
                        default=1,
                        help='The data parallel way.')
    parser.add_argument('--mp',
                        required=False,
                        type=int,
                        default=1,
                        help='The model parallel way.')
    args_opt = parser.parse_args()

    if args_opt.distribute == 'true':
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
            full_batch=True, loss_repeated_mean=True,
            device_num=device_num, enable_parallel_optimizer=False)

    parallel_config = TransformerOpParallelConfig(pipeline_stage=args_opt.pipeline_stage,
                                                  micro_batch_num=args_opt.micro_batch_num,
                                                  model_parallel=args_opt.mp,
                                                  data_parallel=args_opt.dp,
                                                  optimizer_shard=False)

    net = Net(batch=args_opt.batch_size // args_opt.micro_batch_num if args_opt.pipeline_stage else args_opt.batch_size,
              src_len=args_opt.src_len, tgt_len=args_opt.tgt_len,
              vocab_size=args_opt.vocab_size,
              hidden_size=args_opt.d_model,
              en_layer=args_opt.encoder_layer,
              de_layer=args_opt.decoder_layer,
              parallel_config=parallel_config, return_loss=args_opt.train)

    tokenizer = Tokenzier()
    task = ToyDataset(file_path=args_opt.file_path,
                      tokenizer=tokenizer,
                      seq_length=(args_opt.src_len, args_opt.tgt_len))
    dataset = task.get_dataset(batch_size=args_opt.batch_size)

    if args_opt.pipeline_stage > 1:
        net = PipelineCell(net, args_opt.micro_batch_num)
        param = net.infer_param_pipeline_stage()
        print(f"params is:{param}", flush=True)
        group_params = set_weight_decay(param)
        opt = AdamWeightDecay(group_params, learning_rate=args_opt.lr)
    else:
        group_params = set_weight_decay(net.trainable_params())
        opt = AdamWeightDecay(group_params, learning_rate=args_opt.lr)

    if not args_opt.train:
        model = Model(net)
    else:
        model = Model(net, optimizer=opt)

    callback_size = 1
    # single vs pieplien (save a slice of the model)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=callback_size, keep_checkpoint_max=4,
                                   integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="test",
                                 config=ckpt_config)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size), ckpoint_cb]
    model.train(1, dataset, callbacks=callback, dataset_sink_mode=False)


if __name__ == "__main__":
    main()
