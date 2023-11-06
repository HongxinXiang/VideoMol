import timm


def get_timm_models(model_name, pretrained=False, checkpoint_path='', model_params_dict=None):
    """
        find model list: https://github.com/rwightman/pytorch-image-models/blob/475ecdfa3d369b6d482287f2467ce101ce5c276c/results/results-imagenet-a-clean.csv
    :param model_name: e.g. swinv2_large_window12to24_192to384_22kft1k
    :param pretrained: True or False
    :param checkpoint_path:
    :param model_params_dict: params are specified by model, e.g. {'img_size': 224, 'in_chans': 3, 'attn_drop_rate': 0.0, 'drop_rate': 0.0}
        you can find more parameters in https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/
        e.g. SwinTransformer(img_size, patch_size, ...) in swin_transformer.py
    :return: model
    """
    if model_params_dict is not None:
        return timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_params_dict)
    else:
        return timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

