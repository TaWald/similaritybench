cnt = 0
for arch in ["ResNet34", "ResNet101"]:  # , "ResNet101"]:
    trans_pos = [[1]]
    group_id = [0, 1, 2, 3, 4]
    for tp in trans_pos:
        for gid in group_id:  # , 6]:
            for sim_dis_loss in ["AdaptiveDiversityPromotion"]:
                sim_loss = sim_dis_loss[0]
                dis_loss = sim_dis_loss[1]
                dis_loss_weight = "2.00-0.50"  # [0.1, 0.5, 1.0, 2.0, 4.0, 8.0] #
                sim_loss_weight = 1.0
                for dl in dis_loss_weight:  # [01.0]
                    epochs_before_regularization = -1
                    exp_name = "SCIS23"
                    dataset = ["CIFAR10", "CIFAR100"]  # , "ImageNet100"]
                    if arch == "ResNet101":
                        min_gmem = "14.5"
                    else:
                        min_gmem = "10.5"
                    tr_n_models = 1
                    tk = 1
                    td = 1
                    for ds in dataset:  # trans_depth = 1
                        reg_pos = " ".join([str(t) for t in tp])
                        print(
                            f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                            + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                            + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem={min_gmem}G -q gpu"""
                            + f""" ./ke_training.sh"""
                            + f""" ~/Code/knowledge_extension/scripts/training_starts/ke_train_output_regularization.py"""
                            + f""" -exp {exp_name} -d {ds} -a {arch} -pt 0 -fz 0"""
                            + f""" --dis_loss {dis_loss} --dis_loss_weight {dl} -ce 1.0"""
                            + f""" -sm 0 -tr_n_models {tr_n_models} -gid {gid} -na 1 """
                            + f""" -ebr -1"""
                        )
                        cnt += 1
                        if cnt % 4 == 0:
                            print("\n")
