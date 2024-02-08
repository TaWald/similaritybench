from typing import Type

import torchvision.models
from vision.arch import abstract_acti_extr
from vision.util import data_structs as ds
from torch import nn

# from ke.arch import o2o_average

# from ke.arch.ensembling import toworkon_loo_replacement

# Instead of automatically going through the registered subclasses
#   this is explicitly stated to keep the code more readable and traceable.


def get_base_arch(
    arch: ds.BaseArchitecture | str,
) -> Type[abstract_acti_extr.AbsActiExtrArch]:
    """Finds a Model network by its name.
    Should the class not be found it will raise an Error.
        :param arch: Name of the network class that should be used.
    :raises NotImplementedError: If not subclass is found for the given name.
    :return: Subclass with same name as network_name
    """
    from vision.arch import vgg, resnet

    if isinstance(arch, str):
        arch = ds.BaseArchitecture(arch)

    if arch == ds.BaseArchitecture.VGG16:
        return vgg.VGG16
    elif arch == ds.BaseArchitecture.VGG11:
        return vgg.VGG11
    elif arch == ds.BaseArchitecture.VGG19:
        return vgg.VGG19
    elif arch == ds.BaseArchitecture.RESNET18:
        return resnet.ResNet18
    elif arch == ds.BaseArchitecture.RESNET34:
        return resnet.ResNet34
    elif arch == ds.BaseArchitecture.RESNET50:
        return resnet.ResNet50
    elif arch == ds.BaseArchitecture.RESNET101:
        return resnet.ResNet101
    else:
        raise ValueError("Seems like the BaseArchitecture was not added here!")


def get_tv_arch(arch: ds.BasicPretrainableArchitectures, pretrained: bool, n_cls) -> nn.Module:
    if arch == ds.BasicPretrainableArchitectures.TV_VGG11:
        if pretrained:
            model = torchvision.models.vgg11_bn(weights=torchvision.models.VGG11_BN_Weights)
        else:
            model = torchvision.models.vgg11_bn(None)
        model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(512, out_features=n_cls))
        return model
    elif arch == ds.BasicPretrainableArchitectures.TV_VGG16:
        if pretrained:
            model = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights)
        else:
            model = torchvision.models.vgg16_bn(None)
        model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(512, out_features=n_cls))
        return model
    elif arch == ds.BasicPretrainableArchitectures.TV_VGG19:
        if pretrained:
            model = torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights)
        else:
            model = torchvision.models.vgg19_bn(None)
        model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(512, out_features=n_cls))
        return model
    elif arch == ds.BasicPretrainableArchitectures.TV_RESNET18:
        if pretrained:
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights)
        else:
            model = torchvision.models.resnet18(None)
        model.conv1 = nn.Conv2d(3, 64, (3, 3), padding=1, bias=False)
        model.fc = nn.Linear(512, out_features=n_cls)
        return model
    elif arch == ds.BasicPretrainableArchitectures.TV_RESNET34:
        if pretrained:
            model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights)
        else:
            model = torchvision.models.resnet34(None)
        model.conv1 = nn.Conv2d(3, 64, (3, 3), padding=1, bias=False)
        model.fc = nn.Linear(512, out_features=n_cls)
        return model
    else:
        raise ValueError("Seems like the BaseArchitecture was not added here!")
