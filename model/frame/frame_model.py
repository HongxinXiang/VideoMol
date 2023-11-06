import torch
from torch import nn
from model.frame.frame_utils import get_timm_models
from timm.models.swin_transformer import default_cfgs
from model.model_utils import weights_init
from model.model_utils import get_classifier
import torchvision


def SwinTransformer(model_name, only_feat=True, pretrained=False, checkpoint_path='',
                    img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg',
                    embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), head_dim=None,
                    window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                    drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, weight_init=''):
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/swin_transformer.py
    assert model_name in list(default_cfgs.keys()), "{} is not in {}".format(model_name, list(default_cfgs.keys()))
    model_params_dict = {
        "img_size": img_size,
        # "patch_size": patch_size,
        "in_chans": in_chans,
        "num_classes": num_classes,
        "global_pool": global_pool,
        # "embed_dim": embed_dim,
        # "depths": depths,
        # "num_heads": num_heads,
        "head_dim": head_dim,
        # "window_size": window_size,
        "mlp_ratio": mlp_ratio,
        "qkv_bias": qkv_bias,
        "drop_rate": drop_rate,
        "attn_drop_rate": attn_drop_rate,
        "drop_path_rate": drop_path_rate,
        "norm_layer": norm_layer,
        "ape": ape,
        "patch_norm": patch_norm,
        "weight_init": weight_init
    }
    model = get_timm_models(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, model_params_dict=model_params_dict)
    if only_feat:
        model.head = torch.nn.Identity()
    else:  # finetuning network
        model.head
    return model


def get_resnet(modelname, pretrained=False, num_classes=None, ret_num_features=False):
    if modelname == "ResNet18":
        # model = resnet.resnet18(pretrained=False)
        model = torchvision.models.resnet18(pretrained=pretrained)

    if modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=pretrained)

    if modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=pretrained)

    if modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=pretrained)

    if modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes) if num_classes is not None else torch.nn.Identity()

    if ret_num_features:
        return model, num_features
    else:
        return model

# Axis classifier: predicts the x-axis, y-axis or z-axis rotation relationship between two frames
class AxisClassifier(nn.Module):
    def __init__(self, in_features):
        super(AxisClassifier, self).__init__()
        self.axisClassifier = get_classifier(arch="arch1", in_features=in_features, num_tasks=3)

        self.apply(weights_init)

    def forward(self, x):
        logit = self.axisClassifier(x)
        return logit


# Rotation classifier: predict clockwise or counterclockwise rotation between two frames
class RotationClassifier(nn.Module):
    def __init__(self, in_features):
        super(RotationClassifier, self).__init__()
        self.rotationClassifier = get_classifier(arch="arch1", in_features=in_features, num_tasks=2)
        self.apply(weights_init)

    def forward(self, x):
        logit = self.rotationClassifier(x)
        return logit


# Angle classifier: predict the angle difference between two frames (the difference in the rotation angle of the x, y, z axes)
class AngleClassifier(nn.Module):
    def __init__(self, in_features, num_tasks):
        super(AngleClassifier, self).__init__()
        self.angleClassifier = get_classifier(arch="arch1", in_features=in_features, num_tasks=num_tasks)
        self.apply(weights_init)

    def forward(self, x):
        logit = self.angleClassifier(x)
        return logit


# Chemical classifier: predicting their chemical information
class ChemicalClassifier(nn.Module):
    def __init__(self, in_features, num_tasks):
        super(ChemicalClassifier, self).__init__()
        self.chemicalClassifier = get_classifier(arch="arch1", in_features=in_features, num_tasks=num_tasks)
        self.apply(weights_init)

    def forward(self, x):
        logit = self.chemicalClassifier(x)
        return logit


if __name__ == '__main__':
    # model = SwinTransformer(model_name="swinv2_large_window12to24_192to384_22kft1k")
    # print(model)

    batch_size = 8
    in_features = 512
    n_frame = 20
    frame1 = torch.rand(size=(batch_size, in_features))
    frame2 = torch.rand(size=(batch_size, in_features))
    frame = torch.cat([frame1, frame2], dim=1)

    axisClassifier = AxisClassifier(in_features=in_features * 2)
    rotationClassifier = RotationClassifier(in_features=in_features * 2)
    angleClassifier = AngleClassifier(in_features=in_features * 2, num_tasks=n_frame//3-1)

    output_axis = axisClassifier(frame)
    output_rotation = rotationClassifier(frame)
    output_angle = angleClassifier(frame)

    print(123)