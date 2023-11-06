import timm
import torch

from model.base.base_utils import get_timm_model_names, get_predictor


class FramePredictor(torch.nn.Module):
    def __init__(self, model_name, head_arch, num_tasks, pretrained=False, head_arch_params=None, **kwargs):
        super(FramePredictor, self).__init__()

        assert model_name in get_timm_model_names()

        self.model_name = model_name
        self.head_arch = head_arch
        self.num_tasks = num_tasks
        self.pretrained = pretrained
        if head_arch_params is None:
            head_arch_params = {"inner_dim": None, "dropout": 0.2, "activation_fn": None}
        self.head_arch_params = head_arch_params

        # create base model
        self.model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        # some attributes of base model
        self.classifier_name = self.model.default_cfg["classifier"]
        self.in_features = self.get_in_features()
        # self-defined head for prediction
        self_defined_head = self.create_self_defined_head()
        self.set_self_defined_head(self_defined_head)

    def forward(self, x):
        return self.model(x)

    def get_in_features(self):
        if type(self.classifier_name) == str:
            if "." not in self.classifier_name and isinstance(getattr(self.model, self.classifier_name), torch.nn.modules.linear.Identity):
                in_features = self.model.num_features
            else:
                classifier = self.model
                for item in self.classifier_name.split("."):
                    classifier = getattr(classifier, item)
                in_features = classifier.in_features
        elif type(self.classifier_name) == tuple or type(self.classifier_name) == list:
            in_features = []
            for item_name in self.classifier_name:
                classifier = self.model
                for item in item_name.split("."):
                    classifier = getattr(classifier, item)
                in_features.append(classifier.in_features)
        else:
            raise Exception("{} is undefined.".format(self.classifier_name))
        return in_features

    def create_self_defined_head(self):
        if type(self.in_features) == list or type(self.in_features) == tuple:
            assert len(self.classifier_name) == len(self.in_features)
            head_predictor = []
            for item_in_features in self.in_features:
                single_predictor = get_predictor(arch=self.head_arch, in_features=item_in_features,
                                                 num_tasks=self.num_tasks,
                                                 inner_dim=self.head_arch_params["inner_dim"],
                                                 dropout=self.head_arch_params["dropout"],
                                                 activation_fn=self.head_arch_params["activation_fn"])
                head_predictor.append(single_predictor)
        elif type(self.classifier_name) == str:
            head_predictor = get_predictor(arch=self.head_arch, in_features=self.in_features, num_tasks=self.num_tasks,
                                           inner_dim=self.head_arch_params["inner_dim"],
                                           dropout=self.head_arch_params["dropout"],
                                           activation_fn=self.head_arch_params["activation_fn"])
        else:
            raise Exception("error type in classifier_name ({}) and in_features ({})".format(type(self.classifier_name),
                                                                                             type(self.in_features)))
        return head_predictor

    def set_self_defined_head(self, self_defined_head):
        if type(self.classifier_name) == list or type(self.classifier_name) == tuple:
            for predictor_idx, item_classifier_name in enumerate(self.classifier_name):
                classifier = self.model
                if "." in item_classifier_name:
                    split_classifier_name = item_classifier_name.split(".")
                    for i, item in enumerate(split_classifier_name):
                        classifier = getattr(classifier, item)
                        if i == len(split_classifier_name) - 2:
                            setattr(classifier, split_classifier_name[-1], self_defined_head[predictor_idx])
                else:
                    setattr(self.model, item_classifier_name, self_defined_head[predictor_idx])
        elif "." in self.classifier_name:
            classifier = self.model
            split_classifier_name = self.classifier_name.split(".")
            for i, item in enumerate(split_classifier_name):
                classifier = getattr(classifier, item)
                if i == len(split_classifier_name) - 2:
                    setattr(classifier, split_classifier_name[-1], self_defined_head)
        else:
            setattr(self.model, self.classifier_name, self_defined_head)


class RegMeanStdWrapper(torch.nn.Module):
    def __init__(self, model, stat_mean, stat_std, device, evaluate=False):
        super(RegMeanStdWrapper, self).__init__()

        self.model = model
        self.targets_mean = torch.Tensor([stat_mean]).to(device)
        self.targets_std = torch.Tensor([stat_std]).to(device)
        self.evaluate = evaluate

    def forward(self, x):
        reg_output = self.model(x)
        if self.evaluate:
            reg_output = reg_output * self.targets_std + self.targets_mean
        return reg_output

    def set_eval(self):
        self.evaluate = True

    def set_train(self):
        self.evaluate = False