layer_DIFF_tdepth_1_expvar_1 = {
    f"layer_{i}_tdepth_1_expvar_1": {
        "exp_name": "SCIS23",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "dis_loss_weight": 1.00,
        "ce_loss_weight": 1.00,
        "aggregate_reps": True,
        "softmax": True,
        "epochs_before_regularization": -1,
    }
    for i in [1, 3, 5, 7, 9, 11, 13, 15]
}

layer_SPARSEDIFF_tdepth_1_expvar_1 = {
    f"layer_{i}_tdepth_1_expvar_1": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "dis_loss_weight": 1.00,
        "ce_loss_weight": 1.00,
        "aggregate_reps": True,
        "softmax": True,
        "epochs_before_regularization": 0,
    }
    for i in [1, 5, 9, 13, 15]
}

layer_9_tdepth_9_lincka_DIFF = {
    f"layer_9_tdepth_9_lincka_{i}": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [9],
        "trans_depth": 9,
        "kernel_width": 1,
        "sim_loss": "LinCKA",
        "sim_loss_weight": 1.00,
        "dis_loss": "LinCKA",
        "dis_loss_weight": float(i),
        "ce_loss_weight": 1.00,
        "aggregate_reps": True,
        "softmax": True,
        "epochs_before_regularization": 0,
    }
    for i in [0.1, 1.0, 2.0, 4.0, 8.0, 10.0, 12.0]
}
layer_9_tdepth_9_expvar_DIFF = {
    f"layer_9_tdepth_9_expvar_{i}": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [9],
        "trans_depth": 9,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "dis_loss_weight": float(i),
        "ce_loss_weight": 1.00,
        "aggregate_reps": True,
        "softmax": True,
        "epochs_before_regularization": 0,
    }
    for i in [1, 2, 4, 6, 8, 10, 12]
}
layer_DIFF_tdepth_9_expvar_1 = {
    f"layer_{i}_tdepth_9_expvar_1": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 9,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "dis_loss_weight": 1.00,
        "ce_loss_weight": 1.00,
        "aggregate_reps": True,
        "softmax": True,
        "epochs_before_regularization": 0,
    }
    for i in [1, 3, 5, 7, 9, 11, 13, 15, 16]
}


layer_9_tdepth_DIFF_expvar_1 = {
    f"layer_9_tdepth_{i}_expvar_1": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [9],
        "trans_depth": i,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for i in [1, 3, 5, 7, 9]
}

layer_8_tdepth_1_expvar_1 = {
    "layer_8_tdepth_1_expvar_1": {
        "exp_name": "SCIS23",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [8],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
}

layer_9_tdepth_1_expvar_1 = {
    "layer_9_tdepth_1_expvar_1": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [9],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
}

layer_MULTI_tdepth_9_expvar_1 = {
    f"layer_{min(h)}to{max(h)}_tdepth_9_expvar_1": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": list(h),
        "trans_depth": 9,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for h in [[9], range(7, 12), range(5, 14), range(3, 16), range(0, 17)]
}

layer_SCIS23_MULTI_tdepth_1_expvar_1 = {
    f"layer_{min(h)}to{max(h)}_tdepth_1_expvar_1": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": list(h),
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for h in [
        [8],
        list(range(6, 11)),
        list(range(4, 13)),
        list(range(2, 15)),
        list(range(17)),
    ]
}

different_metrics_scis23 = {
    f"diff_metrics_layer_{h[0]}_tdepth_1_{metric}_{dis_loss}": {
        "exp_name": "SCIS23_metric_experiments",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": list(h),
        "trans_depth": 1 if metric != "LinCKA" else 0,
        "kernel_width": 1,
        "sim_loss": metric,
        "sim_loss_weight": 1.00,
        "dis_loss": metric,
        "ce_loss_weight": 1.00,
        "dis_loss_weight": float(dis_loss),
    }
    for h in [[1], [3], [8], [13]]
    for metric in ["ExpVar", "LinCKA", "L2Corr"]
    for dis_loss in ["0.25", "1.00", "4.00"]
}

scis_baseline_metrics_scis23 = {}

scis_baseline_ensembles = {
    "scis23_baseline_ensemble": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [8],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "dis_loss": "None",
    }
}

scis_baseline_ensembles_first_two = {
    "scis23_baseline_ensemble_first_two": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [8],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "dis_loss": "None",
    }
}

scis_exp_var_example_ensemble = {
    "expvar_5_models_layer_8": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [8],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
}

scis_example_ensemble = {
    "lincka_5_models_layer_8": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [8],
        "trans_depth": 0,
        "kernel_width": 1,
        "sim_loss": "LinCKA",
        "sim_loss_weight": 1.00,
        "dis_loss": "LinCKA",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
}


lincka_ensemble_layer_DIFF = {
    f"lincka_5_models_layer_{i}": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 0,
        "kernel_width": 1,
        "sim_loss": "LinCKA",
        "sim_loss_weight": 1.00,
        "dis_loss": "LinCKA",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for i in [1, 3, 8, 13]
}

lincka_ensemble_layer_IN100_DIFF = {
    f"lincka_5_models_layer_{i}": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "ImageNet100",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 0,
        "kernel_width": 1,
        "sim_loss": "LinCKA",
        "sim_loss_weight": 1.00,
        "dis_loss": "LinCKA",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for i in [1, 3, 8, 13]
}

expvar_ensemble_layer_IN100_DIFF = {
    f"lincka_5_models_layer_{i}": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "ImageNet100",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for i in [1, 3, 8, 13]
}

baseline_cifar100_resnet101_layer_diff = {
    f"baseline_cifar100_resnet101": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR100",
        "architecture": "ResNet101",
        "hooks": [8],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "None",
    }
}

lincka_ensemble_cifar100_resnet101_layer_diff = {
    f"lincka_cifar100_resnet101_layer{i}": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR100",
        "architecture": "ResNet101",
        "hooks": [i],
        "trans_depth": 0,
        "kernel_width": 1,
        "sim_loss": "LinCKA",
        "sim_loss_weight": 1.00,
        "dis_loss": "LinCKA",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for i in [1, 3, 7, 20, 32]
}

expvar_ensemble_layer_DIFF = {
    f"expvar_5_models_layer_{i}": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for i in [1, 3, 8, 13]
}

l2corr_ensemble_layer_DIFF = {
    f"l2corr_5_models_layer_{i}": {
        "exp_name": "SCIS23_N_Ensembles",
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "L2Corr",
        "sim_loss_weight": 1.00,
        "dis_loss": "L2Corr",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
    for i in [1, 3, 8, 13]
}


lin_cka_var_5_models_layer_8 = {
    "lin_cka_5_models": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [8],
        "trans_depth": 9,
        "kernel_width": 3,
        "sim_loss": "LinCKA",
        "sim_loss_weight": 1.00,
        "dis_loss": "LinCKA",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
}

l2corr_var_5_models_layer_8 = {
    "l2corr_5_models": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        "hooks": [8],
        "trans_depth": 9,
        "kernel_width": 3,
        "sim_loss": "L2Corr",
        "sim_loss_weight": 1.00,
        "dis_loss": "L2Corr",
        "ce_loss_weight": 1.00,
        "dis_loss_weight": 1.00,
    }
}


baseline_cifar100 = {
    "baseline_cifar_100": {
        "dataset": "CIFAR100",
        "architecture": "ResNet18",
        # 'kernel_width': 1.,
        # 'trans_depth': 1,
    }
}

baseline_cifar10 = {
    "baseline_cifar_10": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        #  "hooks": "15",
        #  "trans_depth": 9,  Can be 1,3,5,7,9
        #  "kernel_width": 1,
        # "sim_loss": "ExpVar",
        # "sim_loss_weight": "1.00",
        #  "dis_loss_weight": "1.00",
        #  "ce_loss_weight": "1.00",
        #  "aggregate_reps": True,
        #  "softmax": True,
        #  "epochs_before_regularization": 0,
    }
}

baseline_imagenet100 = {
    "baseline_ImageNet100": {
        "dataset": "ImageNet100",
        "architecture": "ResNet34",
        #  "hooks": "15",
        #  "trans_depth": 9,  Can be 1,3,5,7,9
        #  "kernel_width": 1,
        # "sim_loss": "ExpVar",
        # "sim_loss_weight": "1.00",
        #  "dis_loss_weight": "1.00",
        #  "ce_loss_weight": "1.00",
        #  "aggregate_reps": True,
        #  "softmax": True,
        #  "epochs_before_regularization": 0,
    }
}

baseline_imagenet1k = {
    "baseline_ImageNet1k": {
        "dataset": "ImageNet",
        "architecture": "ResNet34",
        #  "hooks": "15",
        #  "trans_depth": 9,  Can be 1,3,5,7,9
        #  "kernel_width": 1,
        # "sim_loss": "ExpVar",
        # "sim_loss_weight": "1.00",
        #  "dis_loss_weight": "1.00",
        #  "ce_loss_weight": "1.00",
        #  "aggregate_reps": True,
        #  "softmax": True,
        #  "epochs_before_regularization": 0,
    }
}

baseline_5_ensembles_imagenet1k = {
    f"ensemble_5_baseline_imagenet_{i}": {
        "exp_name": "ImageNet_5_ensembles_resnets",
        "dataset": "ImageNet",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "None",
        "dis_loss_weight": 0.00,
    }
    for i in [1]
}

five_ensembles_imagenet1k = {
    f"ensemble_5_baseline_imagenet_{i}_{dlw:.02f}": {
        "dataset": "ImageNet",
        "architecture": "ResNet34",
        "hooks": [i],
        "trans_depth": 1,
        "kernel_width": 1,
        "sim_loss": "ExpVar",
        "sim_loss_weight": 1.00,
        "dis_loss": "ExpVar",
        "dis_loss_weight": dlw,
    }
    for i in [1, 3, 8, 13]
    for dlw in [0.25, 1.0, 4.0]
}
