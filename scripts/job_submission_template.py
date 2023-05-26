from ke.util import name_conventions as nc

arch = "ResNet101"
queue = "verylong"


ce_loss_weight = 1.0
sim_loss_weight = 1.0
cnt = 0

do_calib = False
do_baseline_training = True
do_normal_ke_training = False
do_output_regularization = False
do_adversarial_lense = False
do_adversarial = False
do_functional_adversarial = False

# Do Calibration
if do_calib:
    for dn in [
        nc.KNOWLEDGE_EXTENSION_DIRNAME,
        # nc.KNOWLEDGE_UNUSEABLE_DIRNAME,
        # nc.KNOWLEDGE_ADVERSARIAL_DIRNAME,
        # nc.KE_ADVERSARIAL_LENSE_DIRNAME,
        # nc.KE_OUTPUT_REGULARIZATION_DIRNAME,
    ]:
        n_parallel = 20
        for cnt in range(n_parallel):
            print(
                f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.5G -q gpu-lowprio"""
                + f""" ./knowledge_extension_v3.sh"""
                + f""" ~/Code/knowledge_extension/scripts/post_train_additions/calibrate.py"""
                + f""" --ke_dir_name {dn} --n_parallel {n_parallel} --id {cnt}"""
            )
            if cnt % 4 == 0:
                print("\n")
        # print("\n")

# To start experiments:
# 1. Baseline training [done] -- Original first models
# 2. Train one model (multi seed) have first approximate new one (multiple positions no regularization) [done]
# 3. Train one model (multi seed) have first approximate new one (multiple depths no regularization) [done]
# 4. Train one model (multi seed) have 1st approx. & 2nd dissimilar (multiple positions, single value) [done]
# 4. Train one model (multi seed) have 1st approx. & 2nd dissimilar (single position, multiple depths) [done]
# 5. Train one model (multi seed) have 1st approx. & 2nd dissimilar (single position, single depth, multi weight) [done]
# 6. Choose location and do more models? -->

if do_baseline_training:
    # Dissimilarity training layerwise
    for arch in ["ResNet18", "ResNet101"]:
        if arch == "ResNet18":
            trans_pos = [[6]] #[[2], [4], [6], [8]] #
            group_id = [0, 1, 2, 3, 4]
        elif arch == "ResNet101":
            continue
            trans_pos = [[3]]#, [7], [30], [33]]
            group_id = [0, 1, 2]
        else:
            raise NotImplementedError(f"Unexpected architecture {arch}!")
        for tp in trans_pos:
            for gid in group_id:  # , 6]:
                for sim_dis_loss in [
                    ("L2Corr", "L2Corr"),
                ]:
                    sim_loss = sim_dis_loss[0]
                    dis_loss_weight = [0.] # [0.1, 0.5, 1.0, 2.0, 4.0, 8.0] #
                    for dl in dis_loss_weight:  #  [01.0]
                        epochs_before_regularization = -1
                        exp_name = "test_post_refactoring"
                        dataset = "ImageNet100"
                        tr_n_models = 2
                        tk = 1
                        for td in [1]: # trans_depth = 1
                            reg_pos = " ".join([str(t) for t in tp])
                            print(
                            f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                            + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                            + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=14.5G -q gpu"""
                            + f""" ./ke_training.sh"""
                            + f""" ~/Code/knowledge_extension/scripts/training_starts/ke_train_model.py"""
                            + f""" -exp {exp_name} -d {dataset} -a {arch} -ar 1 -td {td}"""
                            + f""" -tp {reg_pos} -tk {tk} --sim_loss {sim_loss} --ce_loss_weight {ce_loss_weight}"""
                            + f""" --dis_loss L2Corr --dis_loss_weight {dl}"""
                            + f""" -sm 0 -tr_n_models {tr_n_models} -gid {gid} -na 1 --sim_loss_weight {sim_loss_weight}"""
                            + f""" --epochs_before_regularization {epochs_before_regularization}"""
                            + f""" --save_approximation_layers 1"""
                        )
                            cnt += 1
                            if cnt % 4 == 0:
                                print("\n")
        # print("\n")

if do_normal_ke_training:
    # Dissimilarity training layerwise
    for dis_loss_weight in [0.00, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        for tp in [[2], [5], [8]]:
            for gid in [0, 1]:  # , 6]:
                for dis_sim_loss in [
                    ("L2RelRep", "L2RelRep"),
                ]:
                    sim_loss = dis_sim_loss[0]
                    dis_loss = dis_sim_loss[1]
                    if dis_loss_weight == 0.0:
                        dis_loss = "None"
                    exp_name = "test_post_refactoring"
                    architecture = "ResNet34"
                    dataset = "CIFAR10"
                    tr_n_models = 5
                    tk = 3
                    trans_depth = 9
                    epochs_before_regularization = -1

                    reg_pos = " ".join([str(t) for t in tp])
                    print(
                        f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                        + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                        + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.5G -q gpu-lowprio"""
                        + f""" ./ke_training.sh"""
                        + f""" ~/Code/knowledge_extension/scripts/training_starts/ke_train_model.py"""
                        + f""" -exp {exp_name} -d {dataset} -a {architecture} -ar 1 -td {trans_depth}"""
                        + f""" -tp {reg_pos} -tk {tk} --sim_loss {sim_loss} --ce_loss_weight {ce_loss_weight}"""
                        + f""" --dis_loss L2Corr --dis_loss_weight 0.1"""
                        + f""" -sm 0 -tr_n_models {tr_n_models} -gid {gid} -na 1 --sim_loss_weight {sim_loss_weight}"""
                        + f""" --epochs_before_regularization {epochs_before_regularization}"""
                        + f""" --save_approximation_layers 1"""
                    )

            cnt += 1
            if cnt % 4 == 0:
                print("\n")
        # print("\n")
"""
Adaptive diversity promotion:
[
        "0.01-0.002",
        "0.025-0.0675",
        "0.05-0.0135",
        "0.50-0.125",
        "1.00-0.25",
        "2.00-0.50"
    ]
    
EnsembleEntropyMaximization
[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

FocalCosineSimProbability
[ 
    "0.25-0.50",
    "0.25-1.00",
    "0.25-2.00",
    "0.50-0.50",
    "0.50-1.00",
    "0.50-2.00",
    "1.00-0.50",
    "1.00-1.00",
    "1.00-2.00",
    "2.00-0.50",
    "2.00-1.00",
    "2.00-2.00",
    "4.00-0.50",
    "4.00-1.00",
    "4.00-2.00",
]

"""


if do_output_regularization:
    for dissim_weight in [
        "0.25-0.50",
        "0.25-1.00",
        "0.25-2.00",
        "0.50-0.50",
        "0.50-1.00",
        "0.50-2.00",
        "1.00-0.50",
        "1.00-1.00",
        "1.00-2.00",
        "2.00-0.50",
        "2.00-1.00",
        "2.00-2.00",
        "4.00-0.50",
        "4.00-1.00",
        "4.00-2.00",
    ]:
        for dis_loss in ["FocalCosineSimProbability"]:
            for gid in [0, 1]:  # , 6]:
                for pcgrad in ["1"]:
                    exp_name = "baseline"
                    dataset = "CIFAR10"
                    architecture = "ResNet34"
                    tr_n_models = 5
                    print(
                        f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                        + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                        + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.5G -q gpu-lowprio"""
                        + f""" ./ke_training.sh"""
                        + f""" ~/Code/FeatureComparisonV2/scripts/training_starts/ke_train_output_regularization.py"""
                        + f""" -exp {exp_name} -d {dataset} -a {architecture} -gid {gid} """
                        + f"""--dis_loss {dis_loss} --dis_loss_weight {dissim_weight} """
                        + f""" -sm 0 -tr_n_models {tr_n_models} -na 1 -ebr -1 --pc_grad {pcgrad}"""
                    )

                cnt += 1
                if cnt % 4 == 0:
                    print("\n")
            # print("\n")


if do_adversarial_lense:
    for adv_weight in [0.1, 0.05, 0.01, 0.005, 0.001]:
        for dis_loss in ["AdaptiveDiversityPromotion"]:
            for gid in [0, 1]:  # , 6]:
                exp_name = "first_experiments"
                dataset = "CIFAR10"
                architecture = "ResNet34"
                lense_size = "medium"
                tr_n_models = 5
                reco_weight = 1.0
                print(
                    f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                    + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                    + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.5G -q gpu-lowprio"""
                    + f""" ./ke_training.sh"""
                    + f""" ~/Code/FeatureComparisonV2/scripts/training_starts/ke_train_adversarial_lense.py"""
                    + f""" -exp {exp_name} -d {dataset} -a {architecture} -gid {gid} """
                    + f"""--lense_reco_weight {reco_weight} --lense_adversarial_weight {adv_weight} """
                    + f""" --lense_setting {lense_size} -tr_n_models {tr_n_models} -na 1"""
                )
                cnt += 1
                if cnt % 4 == 0:
                    print("\n")
            # print("\n")


if do_adversarial:
    # Dissimilarity training layerwise
    for dissim_weight in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        for tp in [[8], [16]]:
            for gid in [0, 1]:  # , 6]:
                for grs in [1]:
                    tk = 3
                    tr_n_models = 5
                    trans_depth = 9
                    exp_name = "test_adversarial"
                    architecture = "ResNet34"
                    adv_loss = "ExpVar"
                    dataset = "CIFAR10"
                    mem_demand = "10.6G"
                    reg_pos = " ".join([str(t) for t in tp])
                    epochs_before_regularization = -1

                    print(
                        f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                        + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                        + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem={mem_demand} -q gpu-lowprio"""
                        + f""" ./ke_training.sh"""
                        + f""" ./Code/FeatureComparisonV2/scripts/training_starts/ke_train_adversarial.py"""
                        + f""" -exp {exp_name} -d {dataset} -a {architecture} -ar 1 -td {trans_depth}"""
                        + f""" -tp {reg_pos} -tk {tk} --adv_loss {adv_loss}"""
                        + f""" --adversarial_loss_weight {dissim_weight}"""
                        + f""" --gradient_reversal_scaling {grs}"""
                        + f""" -tr_n_models {tr_n_models} -gid {gid} -na 1"""
                        + f""" --epochs_before_regularization {epochs_before_regularization}"""
                        + f""" --save_approximation_layers 1"""
                    )
                    cnt += 1
                    if cnt % 4 == 0:
                        print("\n")

if do_functional_adversarial:
    # Dissimilarity training layerwise
    for trans_loss_weight in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        for trans_depth in [9]:  # , 3, 5, 7, 9]: # , 3, 5, 7, 9]:  #3, 5, 7, 9]:
            for tp in [[8], [14]]:
                for gid in [0, 1]:  # , 6]:
                    for ce_loss_weight in [1]:
                        for grs in [1]:
                            tk = 3
                            tr_n_models = 5
                            exp_name = "test_adversarial_functional"
                            architecture = "ResNet34"
                            dataset = "CIFAR10"
                            mem_demand = "10.6G"
                            reg_pos = " ".join([str(t) for t in tp])
                            epochs_before_regularization = -1

                            print(
                                f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                                + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                                + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem={mem_demand}"""
                                + f""" -q gpu-lowprio ./ke_training.sh"""
                                + f""" ./Code/FeatureComparisonV2/scripts/training_starts/ke_train_unuseable_downstream_model.py"""
                                + f""" -exp {exp_name} -d {dataset} -a {architecture} -ar 1 -td {trans_depth}"""
                                + f""" -tp {reg_pos} -tk {tk}"""
                                + f""" --transfer_loss_weight {trans_loss_weight}"""
                                + f""" --gradient_reversal_scaling {grs}"""
                                + f""" -tr_n_models {tr_n_models} -gid {gid} -na 1"""
                                + f""" --epochs_before_regularization {epochs_before_regularization}"""
                                + f""" --save_approximation_layers 1"""
                            )
                            cnt += 1
                            if cnt % 4 == 0:
                                print("\n")


if False:
    # Dissimilarity training layerwise
    for dissim_weight in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        for trans_depth in [9]:  # , 3, 5, 7, 9]: # , 3, 5, 7, 9]:  #3, 5, 7, 9]:
            for tp in [[8], [16]]:
                if dissim_weight == 0.0:
                    sim_dis_loss = [("ExpVar", "None")]
                else:
                    sim_dis_loss = [("ExpVar")]
                for sim_loss, dis_loss in sim_dis_loss:  # ,
                    for gid in [0]:  # , 6]:
                        tk = 3
                        tr_n_models = 5
                        exp_name = "test_adversarial"
                        architecture = "ResNet34"
                        dataset = "CIFAR10"
                        mem_demand = "14.0G"
                        reg_pos = " ".join([str(t) for t in tp])
                        print(
                            f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                            + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                            + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem={mem_demand} -q gpu-lowprio"""
                            + f""" ./ke_training.sh ./Code/FeatureComparisonV2/scripts/ke_train_model.py"""
                            + f""" -exp {exp_name} -d {dataset} -a {architecture} -ar 1 -td {trans_depth}"""
                            + f""" -tp {reg_pos} -tk {tk} --adversarial_loss {sim_loss} --dis_loss {dis_loss}"""
                            + f""" --ce_loss_weight {ce_loss_weight} --dis_loss_weight {dissim_weight} -sm 0"""
                            + f""" -tr_n_models {tr_n_models} -gid {gid} -na 1"""
                            + f""" --sim_loss_weight {sim_loss_weight}"""
                            + f""" --epochs_before_regularization {epochs_before_regularization}"""
                            + f""" --save_approximation_layers 1"""
                        )
                        cnt += 1
                        if cnt % 4 == 0:
                            print("\n")

#### Eval trained models
if False:
    print()
    for cnt, ppct in enumerate(unstructured_prune_pct if pt == unstructured_pt else structured_prune_pct):
        if cnt % 5 == 0:
            print()
        print(
            f"""bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=11.G -q gpu-lowprio /home/t006d/feature_comp_v2.sh /home/t006d/Code/FeatureComparisonV2/scripts/eval_trained_models.py -exp exp_nips_22 -d CIFAR10 -pt {pt} -a {arch} -p_pct {str(ppct)}"""
        )
    print(
        f"""bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=11.G -q gpu-lowprio /home/t006d/feature_comp_v2.sh /home/t006d/Code/FeatureComparisonV2/scripts/eval_trained_models.py -exp exp_nips_22 -d CIFAR10 -a {arch}"""
    )
