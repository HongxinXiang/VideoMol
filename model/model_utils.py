import logging
import os
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


# initializing weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_classifier(arch, in_features, num_tasks):
    if arch == "arch1":
        return nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features, in_features // 2)),
            ("Softplus", nn.Softplus()),
            ("linear2", nn.Linear(in_features // 2, num_tasks))
        ]))
    elif arch == "arch2":
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features, 128)),
            ('leakyreLU', nn.LeakyReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('linear2', nn.Linear(128, num_tasks))
        ]))
    elif arch == "arch3":
        return nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features, num_tasks))
        ]))


def save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict, desc, epoch, save_path, name_pre, name_post='_best', logger=None):
    log = print if logger is None else logger.info

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

    try:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            log("Directory {} is created.".format(save_path))
    except:
        pass

    filename = '{}/{}{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    log('model has been saved as {}'.format(filename))


def load_checkpoint(pretrained_pth, videoMol, axisClassifier, rotationClassifier, angleClassifier, chemicalClassifier,
                    optimizer=None, lr_scheduler=None, logger=None):
    log = logger.info if logger is not None else print
    flag = False
    resume_desc = None
    if os.path.isfile(pretrained_pth):
        pretrained_model = torch.load(pretrained_pth)
        resume_desc = pretrained_model["desc"]
        model_list = [("videoMol", videoMol, "videoMol"), ("axisClassifier", axisClassifier, "axisClassifier"),
                      ("rotationClassifier", rotationClassifier, "rotationClassifier"),
                      ("angleClassifier", angleClassifier, "angleClassifier"),
                      ("chemicalClassifier", chemicalClassifier, "chemicalClassifier")]
        if optimizer is not None:
            model_list.append(("optimizer", optimizer, "optimizer"))
        if lr_scheduler is not None:
            model_list.append(("lr_scheduler", lr_scheduler, "lr_scheduler"))
        for name, model, model_key in model_list:
            try:
                model.load_state_dict(pretrained_model[model_key])
            except:
                ckp_keys = list(pretrained_model[model_key])
                cur_keys = list(model.state_dict())
                model_sd = model.state_dict()
                for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                    model_sd[cur_key] = pretrained_model[model_key][ckp_key]
                model.load_state_dict(model_sd)
            log("[resume info] resume {} completed.".format(name))
        flag = True
    else:
        log("===> No checkpoint found at '{}'".format(pretrained_pth))

    return flag, resume_desc


def load_pretrained_component(model, pretrained_pth, model_key, consistency=True, logger=None):
    log = logger if logger is not None else logging
    flag = False  # load successfully when only flag is true
    desc = None
    if pretrained_pth:
        if os.path.isfile(pretrained_pth):
            log.info("===> Loading checkpoint '{}'".format(pretrained_pth))
            checkpoint = torch.load(pretrained_pth, map_location='cpu')
            # load parameters
            ckpt_model_state_dict = checkpoint[model_key]
            if consistency:  # model and ckpt_model_state_dict is consistent.
                model.load_state_dict(ckpt_model_state_dict)
                log.info("load all the parameters of pre-trianed model.")
            else:  # load parameter of layer-wise, resnet18 should load 120 layer at head.
                ckp_keys = list(ckpt_model_state_dict)
                cur_keys = list(model.state_dict())
                len_ckp_keys = len(ckp_keys)
                len_cur_keys = len(cur_keys)
                model_sd = model.state_dict()
                for idx in range(min(len_ckp_keys, len_cur_keys)):
                    ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
                    model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
                model.load_state_dict(model_sd)
                log.info("load the first {} parameters. layer number: model({}), pretrain({})"
                         .format(min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))
            desc = "[resume model info] The pretrained_model is at checkpoint {}. \t info: {}"\
                .format(checkpoint['epoch'], checkpoint['desc'])
            flag = True
        else:
            log.info("===> No checkpoint found at '{}'".format(pretrained_pth))
    else:
        log.info('===> No pre-trained model')
    return flag, desc


def load_finetune_component(model, pretrained_pth, model_key, consistency=True, logger=None):
    log = logger if logger is not None else logging
    flag = False  # load successfully when only flag is true
    desc = None
    if pretrained_pth:
        if os.path.isfile(pretrained_pth):
            log.info("===> Loading checkpoint '{}'".format(pretrained_pth))
            checkpoint = torch.load(pretrained_pth, map_location='cpu')
            # load parameters
            ckpt_model_state_dict = checkpoint[model_key]
            if consistency:  # model and ckpt_model_state_dict is consistent.
                model.load_state_dict(ckpt_model_state_dict)
                log.info("load all the parameters of pre-trianed model.")
            else:  # load parameter of layer-wise, resnet18 should load 120 layer at head.
                ckp_keys = list(ckpt_model_state_dict)
                cur_keys = list(model.state_dict())
                len_ckp_keys = len(ckp_keys)
                len_cur_keys = len(cur_keys)
                model_sd = model.state_dict()
                for idx in range(min(len_ckp_keys, len_cur_keys)):
                    ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
                    model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
                model.load_state_dict(model_sd)
                log.info("load the first {} parameters. layer number: model({}), pretrain({})"
                         .format(min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))
            del checkpoint['result_dict']['highest_valid_desc'], checkpoint['result_dict']['final_train_desc'], \
            checkpoint['result_dict']['final_test_desc'], checkpoint['result_dict']['data_split']
            if 'ext_data_dict' in checkpoint['result_dict']:
                del checkpoint['result_dict']['ext_data_dict']
            desc = "[resume model info] The pretrained_model is at checkpoint {}. \t info: {}"\
                .format(checkpoint['epoch'], checkpoint['result_dict'])
            flag = True
        else:
            log.info("===> No checkpoint found at '{}'".format(pretrained_pth))
    else:
        log.info('===> No pre-trained model')
    return flag, desc


def write_result_dict_to_tb(tb_writer: SummaryWriter, result_dict: dict, optimizer_dict: dict, show_epoch=True):
    loop = result_dict["epoch"] if show_epoch else result_dict["step"]
    for key in result_dict.keys():
        if key == "epoch" or key == "step":
            continue
        tb_writer.add_scalar(key, result_dict[key], loop)
    for key in optimizer_dict.keys():
        optimizer = optimizer_dict[key]
        tb_writer.add_scalar(key, optimizer.param_groups[0]["lr"], loop)


