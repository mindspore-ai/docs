"""converter & unit test"""
import numpy as np
import torch
import mindspore as ms
from resnet_ms.src.resnet import resnet50 as ms_resnet50
from resnet_pytorch.resnet import resnet50 as pt_resnet50

np.random.seed(1)


def pytorch_params(pth_file):
    """Get PyTorch parameter"""
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params


def mindspore_params(network):
    """Get MindSpore parameter"""
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params


def param_convert(ms_params, pt_params, ckpt_path):
    """Parameter dictionary mapping"""
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    for ms_param in ms_params.keys():
        if "bn" in ms_param or "downsample.1" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        else:
            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    ms.save_checkpoint(new_params_list, ckpt_path)


def check_res(pth_path, ckpt_path):
    """unit test"""
    inp = np.random.uniform(-1, 1, (4, 3, 224, 224)).astype(np.float32)
    ms_resnet = ms_resnet50(num_classes=10).set_train(False)
    pt_resnet = pt_resnet50(num_classes=10).eval()
    pt_resnet.load_state_dict(torch.load(pth_path, map_location='cpu'))
    ms.load_checkpoint(ckpt_path, ms_resnet)
    print("========= pt_resnet conv1.weight ==========")
    print(pt_resnet.conv1.weight.detach().numpy().reshape((-1,))[:10])
    print("========= ms_resnet conv1.weight ==========")
    print(ms_resnet.conv1.weight.data.asnumpy().reshape((-1,))[:10])
    pt_res = pt_resnet(torch.from_numpy(inp))
    ms_res = ms_resnet(ms.Tensor(inp))
    print("========= pt_resnet res ==========")
    print(pt_res)
    print("========= ms_resnet res ==========")
    print(ms_res)
    print("diff", np.max(np.abs(pt_res.detach().numpy() - ms_res.asnumpy())))


if __name__ == '__main__':
    pt_path = "resnet.pth"
    ms_path = "resnet50.ckpt"
    ms_parameters = mindspore_params(ms_resnet50(num_classes=10))
    print("=====================================")
    pt_parameters = pytorch_params(pt_path)
    param_convert(ms_parameters, pt_parameters, ms_path)

    check_res(pt_path, ms_path)
