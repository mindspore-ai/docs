"""Parallel Optimizer Fusion Example"""
from mindspore.communication import init
from mindspore import context, ParallelMode
from mindspore.nn.transformer import Transformer, TransformerOpParallelConfig

init()
context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, enable_parallel_optimizer=True)
def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):

    # 将transformer的融合层数设置为4个
    gradient_aggregation_group = 4
    dis = max(int((layers + offset) / gradient_aggregation_group), 1)
    # 此处的network是一个cell，用户可以针对自己的网络层调用set_comm_fusion方法
    network.set_comm_fusion(int((layer_id + offset) / dis) + 1)

model_parallel_config = TransformerOpParallelConfig()
net = Transformer(encoder_layers=1, decoder_layers=1,
                  batch_size=4, src_seq_length=10,
                  tgt_seq_length=10, hidden_size=24,
                  num_heads=8, attention_dropout_rate=0.0,
                  hidden_dropout_rate=0.0, lambda_func=set_parallel_configure_for_layer,
                  ffn_hidden_size=24, parallel_config=model_parallel_config)
for item in net.trainable_params():
    if 'dense1' in item.name:
        print(f"The parameter {item.name}'s fusion id is {item.comm_fusion}")
