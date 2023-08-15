from __future__ import annotations

from ke.util import data_structs as ds


def get_default_parameters(architecture_name: str, dataset: ds.Dataset) -> ds.Params:
    if dataset == ds.Dataset.CIFAR10:
        params = ds.Params(
            architecture_name=architecture_name,
            num_epochs=250,  # 250,
            save_last_checkpoint=True,
            batch_size=128,
            label_smoothing=False,
            label_smoothing_val=0.1,
            cosine_annealing=True,
            gamma=0.1,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
            split=0,
            dataset=dataset.value,
        )
    elif dataset == ds.Dataset.DermaMNIST:
        params = ds.Params(
            architecture_name=architecture_name,
            num_epochs=200,  # 250,
            save_last_checkpoint=True,
            batch_size=128,
            label_smoothing=False,
            label_smoothing_val=0.1,
            cosine_annealing=True,
            gamma=0.1,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
            split=0,
            dataset=dataset.value,
        )
    elif dataset in [ds.Dataset.IMAGENET, ds.Dataset.IMAGENET100]:
        """Hyperparams taken from
         https://github.com/tensorflow/tpu/tree/master/models/official/resnet
         /resnet_rs/configs
        For VGG16/19/ResNet34/ResNet50/DenseNet121 the resnetrs50 were used.
        For DenseNet161/ResNet101 the ResNetRs101 is used.
        """
        params = ds.Params(
            architecture_name=architecture_name,
            num_epochs=250,
            save_last_checkpoint=True,
            batch_size=256,
            label_smoothing=True,
            label_smoothing_val=0.1,
            cosine_annealing=True,
            gamma=0.1,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=4e-5,
            split=0,
            dataset=dataset.value,
        )

    elif dataset == ds.Dataset.CIFAR100:
        params = ds.Params(
            architecture_name=architecture_name,
            num_epochs=200,
            save_last_checkpoint=True,
            batch_size=128,
            label_smoothing=False,
            label_smoothing_val=0.1,
            cosine_annealing=True,
            gamma=0.1,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
            split=0,
            dataset=dataset.value,
        )

    elif dataset == ds.Dataset.SPLITCIFAR100:
        params = ds.Params(
            architecture_name=architecture_name,
            num_epochs=1000,
            save_last_checkpoint=True,
            batch_size=128,
            label_smoothing=False,
            label_smoothing_val=0.1,
            cosine_annealing=True,
            gamma=0.1,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
            split=0,
            dataset=dataset.value,
        )
    else:
        raise NotImplementedError(f"Passed Dataset ({dataset}) not implemented.")
    return params


def get_default_arch_params(dataset: ds.Dataset | str) -> dict:
    if isinstance(dataset, str):
        dataset = ds.Dataset(dataset)
    if dataset == ds.Dataset.CIFAR10:
        output_classes = 10
        in_ch = 3
        input_resolution = (32, 32)
        early_downsampling = False
        global_average_pooling = 4
    elif dataset in [ds.Dataset.IMAGENET, ds.Dataset.IMAGENET100]:
        output_classes = 1000 if dataset == ds.Dataset.IMAGENET else 100
        in_ch = 3
        input_resolution = (160, 160)
        early_downsampling = True
        global_average_pooling = 5
    elif dataset == ds.Dataset.CIFAR100:
        output_classes = 100
        in_ch = 3
        input_resolution = (32, 32)
        early_downsampling = False
        global_average_pooling = 4
    elif dataset == ds.Dataset.SPLITCIFAR100:
        output_classes = 5  # 20 Splits a 5 Classes
        in_ch = 3
        input_resolution = (32, 32)
        early_downsampling = False
        global_average_pooling = 4
    elif dataset == ds.Dataset.DermaMNIST:
        output_classes = 7  # 20 Splits a 5 Classes
        in_ch = 3
        input_resolution = (28, 28)
        early_downsampling = False
        global_average_pooling = 4
    else:
        raise NotImplementedError(f"Unexpected dataset! Got {dataset}!")

    return dict(
        n_cls=output_classes,
        in_ch=in_ch,
        input_resolution=input_resolution,
        early_downsampling=early_downsampling,
        global_average_pooling=global_average_pooling,
    )
