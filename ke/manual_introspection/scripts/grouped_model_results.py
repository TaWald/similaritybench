layer_DIFF_tdepth_1_expvar_1 = {
    f"layer_{i}_tdepth_1_ExpVar_1": {
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
    for i in [1, 3, 5, 7, 9, 11, 13, 15]
}

layer_SPARSEDIFF_tdepth_1_expvar_1 = {
    f"layer_{i}_tdepth_1_ExpVar_1": {
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
    f"layer_9_tdepth_9_LinCKA_{i}": {
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
    f"layer_9_tdepth_9_ExpVar_{i}": {
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
    f"layer_9_tdepth_{i}_ExpVar_1": {
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

layer_MULTI_tdepth_9_expvar_1 = {
    f"layer_{min(h)}to{max(h)}_tdepth_9_ExpVar_1": {
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

baseline_unregularized = {
    "baselines": {
        "dataset": "CIFAR10",
        "architecture": "ResNet34",
        #  "hooks": "15",
        #  "trans_depth": 9,  Can be 1,3,5,7,9
        #  "kernel_width": 1,
        "sim_loss": "ExpVar",
        # "sim_loss_weight": "1.00",
        "dis_loss": "None",
        #  "dis_loss_weight": "1.00",
        #  "ce_loss_weight": "1.00",
        #  "aggregate_reps": True,
        #  "softmax": True,
        #  "epochs_before_regularization": 0,
    }
}
