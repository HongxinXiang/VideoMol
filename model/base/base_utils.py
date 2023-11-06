import os
from collections import OrderedDict
from typing import List, Callable

import timm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def get_timm_model_names():
    model_list = timm.list_models()
    return model_list


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "tanh",
        "softplus",
        "linear",
    ]


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "linear":
        return nn.Identity()
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_predictor(arch, in_features, num_tasks, inner_dim=None, dropout=0.2, activation_fn=None):
    if inner_dim is None:
        inner_dim = in_features // 2
    if activation_fn is None:
        activation_fn = "gelu"

    if arch == "arch1":
        return nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features, inner_dim)),
            ("Softplus", nn.Softplus()),
            ("linear2", nn.Linear(inner_dim, num_tasks))
        ]))
    elif arch == "arch2":
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features, 128)),
            ('leakyreLU', nn.LeakyReLU()),
            ('dropout', nn.Dropout(dropout)),
            ('linear2', nn.Linear(128, num_tasks))
        ]))
    elif arch == "arch3":
        return nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features, num_tasks))
        ]))
    elif arch == "arch4":
        return nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout(dropout)),
            ("linear1", nn.Linear(in_features, inner_dim)),
            ("activator", get_activation_fn(activation_fn)),
            ("dropout2", nn.Dropout(dropout)),
            ("linear2", nn.Linear(inner_dim, num_tasks))
        ]))
    elif arch == "none":
        return nn.Identity()


# initializing weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict, desc, epoch, save_path, name_pre, name_post='_best'):
    state = {
        'epoch': epoch,
        'desc': desc
    }

    if model_dict is not None:
        for key in model_dict.keys():
            model = model_dict[key]
            state[key] = {k: v.cpu() for k, v in model.state_dict().items()}
    if optimizer_dict is not None:
        for key in optimizer_dict.keys():
            optimizer = optimizer_dict[key]
            state[key] = optimizer.state_dict()
    if lr_scheduler_dict is not None:
        for key in lr_scheduler_dict.keys():
            lr_scheduler = lr_scheduler_dict[key]
            state[key] = lr_scheduler.state_dict()

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory ", save_path, " is created.")

    filename = '{}/{}{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    print('model has been saved as {}'.format(filename))


def write_result_dict_to_tb(tb_writer: SummaryWriter, result_dict: dict, optimizer_dict: dict, show_epoch=True):
    loop = result_dict["epoch"] if show_epoch else result_dict["step"]
    for key in result_dict.keys():
        if key == "epoch" or key == "step":
            continue
        tb_writer.add_scalar(key, result_dict[key], loop)
    for key in optimizer_dict.keys():
        optimizer = optimizer_dict[key]
        tb_writer.add_scalar(key, optimizer.param_groups[0]["lr"], loop)

