# Arch name to hook number dict.
from decimal import Decimal
from decimal import getcontext

import numpy as np

arch_info = {"VGG19": 16}

dataset = "CIFAR10"
arch = "VGG19"
split = 0
na = 0
exp_name = "basic_celu"
loss = "CELULoss"
transfer_depth = 1
transfer_width = 1
transfer_kernel = 1
group_ids = [0, 1]
n_runs = 8
transfer_layers = [4, 7, 10]  # list(range(16))
getcontext().prec = 2
pct = np.linspace(start=0, stop=1, num=11)
ce_weight, dissim_weight = zip(*[(Decimal(p * 2), Decimal((1 - p) * 2)) for p in pct])

comp = "OLS"

exclude_dgx = """-R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']" """
train_gpu = True
if train_gpu:
    for group_id in group_ids:
        for cnt_layers, tl in enumerate(transfer_layers):
            for cnt_weights, (ce, dissim) in enumerate(zip(ce_weight, dissim_weight)):
                if ce == 0 or dissim == 0:
                    continue
                exp_name = f"HParamCheck--ce-{ce:.01f}--dis-{dissim:.01f}"
                print(
                    """bsub -L /bin/bash  """
                    + """-R "select[hname!='e230-dgx1-1']" -R "tensorcore" """
                    + """-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.5G  -q gpu-lowprio """
                    + f"""./feature_comp_v2.sh ./Code/FeatureComparisonV2/scripts/ke_train_model.py -d {dataset} -a {arch} """
                    + f"""-s {split} -na {na} -exp {exp_name} -td {transfer_depth} -tw {transfer_width} """
                    + f"""-tk {transfer_kernel} -loss {loss} -ce {ce:.01f} -dissim {dissim:.01f} -smr2s {True} -gid {group_id} """
                    + f"""-tl {tl} -tr_n_models {n_runs} -ar {False}"""
                )
                if cnt_weights % 3 == 2:
                    print("\n")
            if cnt_layers % 3 == 2:
                print("\n")


compare_models = False
if compare_models:
    use_gpu_cluster = True
    for group_id in group_ids:
        for cnt_layers, tl in enumerate(transfer_layers):
            for cnt_weights, (ce, dissim) in enumerate(zip(ce_weight, dissim_weight)):
                if ce == 0 or dissim == 0:
                    continue
                exp_name = f"HParamCheck--ce-{ce:.01f}--dis-{dissim:.01f}"
                if use_gpu_cluster:
                    print(
                        """bsub -L /bin/bash  """
                        + """-R "tensorcore" """
                        + """-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.5G -q gpu-lowprio """
                        + """./feature_comp_v2.sh ./Code/FeatureComparisonV2/scripts/ke_compare_models.py """
                        + f"""-exp {exp_name} -td {transfer_depth} -tw {transfer_width} """
                        + f"""-tk {transfer_kernel} -gid {group_id} -tl {tl} -comp {comp}"""
                    )
                else:
                    print(
                        """bsub -n 32 -R "rusage[mem=64GB]" -q verylong """
                        + """./feature_comp_v2.sh ./Code/FeatureComparisonV2/scripts/ke_compare_models.py """
                        + f"""-exp {exp_name} -td {transfer_depth} -tw {transfer_width} """
                        + f"""-tk {transfer_kernel} -gid {group_id} -tl {tl} -comp {comp}"""
                    )
                if cnt_weights % 3 == 2:
                    print("\n")
            if cnt_layers % 3 == 2:
                print("\n")

io_comp = "CohensKappa"
compare_io = False
if compare_io:
    for group_id in group_ids:
        for cnt_layers, tl in enumerate(transfer_layers):
            for cnt_weights, (ce, dissim) in enumerate(zip(ce_weight, dissim_weight)):
                if ce == 0 or dissim == 0:
                    continue
                exp_name = f"HParamCheck--ce-{ce:.01f}--dis-{dissim:.01f}"
                print(
                    """bsub -n 32 -R "rusage[mem=64GB]" -q verylong """
                    + """./feature_comp_v2.sh ./Code/FeatureComparisonV2/scripts/ke_compare_io.py """
                    + f"""-exp {exp_name} -td {transfer_depth} -tw {transfer_width} """
                    + f"""-tk {transfer_kernel} -gid {group_id} -tl {tl} -comp {io_comp}"""
                )
                if cnt_weights % 3 == 2:
                    print("\n")
            if cnt_layers % 3 == 2:
                print("\n")


do_ensemble = False
if do_ensemble:
    for group_id in group_ids:
        for cnt_layers, tl in enumerate(transfer_layers):
            for cnt_weights, (ce, dissim) in enumerate(zip(ce_weight, dissim_weight)):
                if ce == 0 or dissim == 0:
                    continue
                exp_name = f"HParamCheck--ce-{ce:.01f}--dis-{dissim:.01f}"
                print(
                    """bsub -L /bin/bash  """
                    + """-R "tensorcore" """
                    + """-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.5G -q gpu-lowprio """
                    + """./feature_comp_v2.sh ./Code/FeatureComparisonV2/scripts/ke_ensemble_evaluation.py """
                    + f"""-exp {exp_name} -td {transfer_depth} -tw {transfer_width} """
                    + f"""-tk {transfer_kernel} -gid {group_id} -tl {tl}"""
                )
                if cnt_weights % 3 == 2:
                    print("\n")
            if cnt_layers % 3 == 2:
                print("\n")
