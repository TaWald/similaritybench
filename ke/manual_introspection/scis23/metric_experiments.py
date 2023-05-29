
cnt = 0
for arch in ["ResNet34"]:  # , "ResNet101"]:
    trans_pos = [[3], [8], [13]]   # <-- ToDo: Choose appropriate position?
    group_id = [0, 1, 2, 3, 4]
    for tp in trans_pos:
        for gid in group_id:  # , 6]:
            for sim_dis_loss in [
                ("ExpVar", "ExpVar"),
                ("LinCKA", 'LinCKA'),
                ("L2Corr", "L2Corr")
            ]:
                sim_loss = sim_dis_loss[0]
                dis_loss = sim_dis_loss[1]
                dis_loss_weight = [1.00]  # [0.1, 0.5, 1.0, 2.0, 4.0, 8.0] #
                ce_loss_weight = 1.00
                sim_loss_weight = 1.0
                for dl in dis_loss_weight:  # [01.0]
                    epochs_before_regularization = -1
                    exp_name = "SCIS23"
                    dataset = "CIFAR10"
                    tr_n_models = 2
                    tk = 1
                    if sim_loss != 'LinCKA':
                        td = 0
                    else:
                        td = 1

                    reg_pos = " ".join([str(t) for t in tp])
                    print(
                        f"""bsub -L /bin/bash -R "select[hname!='e230-dgx1-1']" """
                        + f""" -R "select[hname!='e230-dgxa100-4']" -R "tensorcore" """
                        + f"""-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=14.5G -q gpu"""
                        + f""" ./ke_training.sh"""
                        + f""" ~/Code/knowledge_extension/scripts/training_starts/ke_train_model.py"""
                        + f""" -exp {exp_name} -d {dataset} -a {arch} -ar 1 -td {td}"""
                        + f""" -tp {reg_pos} -tk {tk} --sim_loss {sim_loss} --ce_loss_weight {ce_loss_weight}"""
                        + f""" --dis_loss {dis_loss} --dis_loss_weight {dl}"""
                        + f""" -sm 0 -tr_n_models {tr_n_models} -gid {gid} -na 1 --sim_loss_weight {sim_loss_weight}"""
                        + f""" --epochs_before_regularization {epochs_before_regularization}"""
                        + f""" --save_approximation_layers 1"""
                    )
                    cnt += 1
                    if cnt % 4 == 0:
                        print("\n")
