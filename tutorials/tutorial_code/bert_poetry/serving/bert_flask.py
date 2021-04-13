import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

from flask import Flask, request, jsonify
import grpc
import numpy as np
import ms_service_pb2
import ms_service_pb2_grpc
import time
import numpy as np
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from src.poetry_dataset import sequence_padding
from src.poetry_dataset import create_tokenizer



app = Flask(__name__)
channel = grpc.insecure_channel('localhost:3300')
stub = ms_service_pb2_grpc.MSServiceStub(channel)
_, tokenizer, _ = create_tokenizer()

def input_construction(request, input_ids, segment_ids, pad_mask):

    input_ids = input_ids.astype(np.int32)
    segment_ids = segment_ids.astype(np.int32)
    pad_mask = pad_mask.astype(np.float32)

    request_input_ids = request.data.add()
    request_input_ids.tensor_shape.dims.extend(list(input_ids.shape))
    request_input_ids.tensor_type = ms_service_pb2.MS_INT32
    request_input_ids.data = input_ids.tobytes()

    request_segment_ids = request.data.add()
    request_segment_ids.tensor_shape.dims.extend(list(segment_ids.shape))
    request_segment_ids.tensor_type = ms_service_pb2.MS_INT32
    request_segment_ids.data = segment_ids.tobytes()

    request_pad_mask = request.data.add()
    request_pad_mask.tensor_shape.dims.extend(list(pad_mask.shape))
    request_pad_mask.tensor_type = ms_service_pb2.MS_FLOAT32
    request_pad_mask.data = pad_mask.tobytes()

    return request, request_input_ids, request_segment_ids, request_pad_mask

def model_predict(stub, request):
    try:
        start_time = time.time()
        result = stub.Predict(request)
        print("time cost is %s" %(time.time()-start_time))
        result_np = np.frombuffer(result.result[0].data, dtype=np.float32).reshape(result.result[0].tensor_shape.dims)
        #print("ms client received: ")
        #print(result_np)
    except grpc.RpcError as e:
        print(e.details())
        status_code = e.code()
        print(status_code.name)
        print(status_code.value)
        exit()
    return result_np

def generate_random_poetry(s, *data):
    token_ids, segment_ids = tokenizer.encode(s)
    token_ids = token_ids[:-1]
    segment_ids = segment_ids[:-1]
    target_ids = []

    stub, request, request_input_ids, request_segment_ids, request_pad_mask = data

    MAX_LEN = 64
    length = 128
    while len(token_ids) + len(target_ids) < MAX_LEN:
        _target_ids = token_ids + target_ids
        _segment_ids = segment_ids + [0 for _ in target_ids]
        index = len(_target_ids)
        _target_ids = sequence_padding(np.array(_target_ids), length=length)
        _segment_ids = sequence_padding(np.array(_segment_ids), length=length)
        pad_mask = (_target_ids != 0).astype(np.float32)
        _target_ids = _target_ids.astype(np.int32)
        _segment_ids = _segment_ids.astype(np.int32)

        request_input_ids.data = _target_ids.tobytes()
        request_segment_ids.data = _segment_ids.tobytes()
        request_pad_mask.data = pad_mask.tobytes()

        _probas = model_predict(stub, request)
        _probas = _probas[0, index-1, 3:]
        p_args = _probas.argsort()[::-1][:100]
        p = _probas[p_args]
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index] + 3
        target_ids.append(target)
        if target == 3:
            break
    poetry = tokenizer.decode(token_ids + target_ids)
    return poetry

def generate_hidden(head, *data):
    token_ids, segment_ids = tokenizer.encode('')
    token_ids = token_ids[:-1]
    segment_ids = segment_ids[:-1]

    punctuations = ['，', '。']
    punctuation_ids = [tokenizer.token_to_id(token) for token in punctuations]
    poetry = []
    length = 128

    stub, request, request_input_ids, request_segment_ids, request_pad_mask = data

    for ch in head:
        poetry.append(ch)
        token_id = tokenizer.token_to_id(ch)
        token_ids.append(token_id)
        segment_ids.append(0)
        while True:
            index = len(token_ids)
            _target_ids = sequence_padding(np.array(token_ids), length=length)
            _segment_ids = sequence_padding(np.array(segment_ids), length=length)

            pad_mask = (_target_ids != 0).astype(np.float32)
            _target_ids = _target_ids.astype(np.int32)
            _segment_ids = _segment_ids.astype(np.int32)

            request_input_ids.data = _target_ids.tobytes()
            request_segment_ids.data = _segment_ids.tobytes()
            request_pad_mask.data = pad_mask.tobytes()

            _probas = model_predict(stub, request)

            _probas = _probas[0, index-1, 3:]
            p_args = _probas.argsort()[::-1][:100]
            p = _probas[p_args]
            p = p / sum(p)
            target_index = np.random.choice(len(p), p=p)
            target = p_args[target_index] + 3
            token_ids.append(target)
            segment_ids.append(0)
            if target > 3:
                poetry.append(tokenizer.id_to_token(target))
            if target in punctuation_ids:
                break
    return ''.join(poetry)

def generate(s='', type=0):
    if len(sys.argv) > 2:
        sys.exit("input error")
    channel_str = ""
    if len(sys.argv) == 2:
        split_args = sys.argv[1].split('=')
        if len(split_args) > 1:
            channel_str = split_args[1]
        else:
            channel_str = 'localhost:5500'
    else:
        channel_str = 'localhost:5500'

    channel = grpc.insecure_channel(channel_str)
    stub = ms_service_pb2_grpc.MSServiceStub(channel)
    request = ms_service_pb2.PredictRequest()

    _target_ids = np.ones(shape=(1,128))
    _segment_ids = np.ones(shape=(1,128))
    pad_mask = np.ones(shape=(1,128))
    request, request_input_ids, request_segment_ids, request_pad_mask = input_construction(request, _target_ids, _segment_ids, pad_mask)
    if type in [0, 1]:
        poetry = generate_random_poetry(s, stub, request, request_input_ids, request_segment_ids, request_pad_mask)
    else:
        poetry = generate_hidden(s, stub, request, request_input_ids, request_segment_ids, request_pad_mask)

    print(poetry)
    return poetry

@app.route('/', methods=['POST'])
def bert():
    if request.method == 'POST':
        s = request.get_json()['string']
        type = request.get_json()['type']
        print("s is {}".format(s))
        print("type is {}".format(type))
        poem = generate(s, type)

    return poem

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080) 
