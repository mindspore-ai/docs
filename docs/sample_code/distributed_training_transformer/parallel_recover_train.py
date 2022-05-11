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
from mindspore.nn.transformer import TransformerOpParallelConfig
from mindspore import Model, ParallelMode, reset_auto_parallel_context, set_auto_parallel_context
import mindspore.communication as D
from mindspore.nn import PipelineCell
from mindspore import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.nn import AdamWeightDecay
from mindspore import load_checkpoint, load_param_into_net
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
    parser.add_argument('--mp',
                        required=False,
                        type=int,
                        default=1,
                        help='The model parallel way.')
    parser.add_argument("--enable_parallel_optimizer",
                        type=int,
                        default=1,
                        choices=[1, 0],
                        help="Enable parallel optimizer, default is enable.")
    parser.add_argument("--ckpt_file",
                        type=str,
                        default="",
                        help="Ckpt files, default is None.")
    args_opt = parser.parse_args()

    if args_opt.distribute == 'true':
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        dp = device_num // args_opt.mp // args_opt.pipeline_stage
        print("rank_id is {}, device_num is {}, dp is {}".format(rank_id, device_num, dp))
        gradient_accumulation_shard = dp > 1 and args_opt.pipeline_stage > 1
        reset_auto_parallel_context()
        set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
            full_batch=True, loss_repeated_mean=True,
            device_num=device_num, enable_parallel_optimizer=bool(args_opt.enable_parallel_optimizer),
            parallel_optimizer_config={"gradient_accumulation_shard": gradient_accumulation_shard})
    else:
        dp = 1

    parallel_config = TransformerOpParallelConfig(pipeline_stage=args_opt.pipeline_stage,
                                                  micro_batch_num=args_opt.micro_batch_num,
                                                  model_parallel=args_opt.mp,
                                                  data_parallel=dp)

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
    # single vs pipeline (save a slice of the model)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=callback_size, keep_checkpoint_max=4,
                                   integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="test",
                                 config=ckpt_config)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size), ckpoint_cb]
    if args_opt.ckpt_file:
        param_dict = load_checkpoint(args_opt.ckpt_file)
        model.build(train_dataset=dataset, epoch=2)
        load_param_into_net(net, param_dict)
    model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)


if __name__ == "__main__":
    main()
