import argparse
import sys
from pathlib import Path

from ke.training.trainers.eval_trainer import EvalTrainer
from ke.util import data_structs as ds
from ke.util import default_parser_args as dpa
from ke.util import file_io
from ke.util import name_conventions as nc
from ke.util.default_params import get_default_parameters
from scripts.post_train_additions.utils import clean_up_after_processing
from scripts.post_train_additions.utils import should_process_a_dir


def main():
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.dir_parser_arguments(parser)
    args = parser.parse_args()
    ke_dirname = args.ke_dir_name

    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

    ke_data_path = base_data_path / ke_dirname
    ke_ckpt_path = base_ckpt_path / ke_dirname

    for res in ke_data_path.iterdir():
        dir_name = res.name

        all_training_infos: list[ds.FirstModelInfo] = []
        kedp = ke_data_path / dir_name
        kecp = ke_ckpt_path / dir_name

        if not should_process_a_dir(kedp):
            continue

        if dir_name.startswith(nc.KE_FIRST_MODEL_DIR.split("__")[0]):
            _, dataset_name, architecture_name = dir_name.split("__")
            p: ds.Params = get_default_parameters(architecture_name, ds.Dataset(dataset_name))
            for sub_dir in res.iterdir():
                if sub_dir.name.startswith("in_progress.txt"):
                    continue
                group_id = int(sub_dir.name.split("_")[-1])
                all_training_infos.append(
                    file_io.get_first_model(
                        ke_data_path=ke_data_path,
                        ke_ckpt_path=ke_ckpt_path,
                        architecture=architecture_name,
                        dataset=dataset_name,
                        learning_rate=p.learning_rate,
                        split=p.split,
                        weight_decay=p.weight_decay,
                        batch_size=p.batch_size,
                        group_id=group_id,
                    )
                )
        else:
            if ke_dirname == nc.KNOWLEDGE_EXTENSION_DIRNAME:
                all_training_infos = file_io.get_trained_ke_models(kedp, kecp)
            elif ke_dirname == nc.KE_OUTPUT_REGULARIZATION_DIRNAME:
                all_training_infos = file_io.get_trained_keo_models(kedp, kecp)
            elif ke_dirname == nc.KE_ADVERSARIAL_LENSE_DIRNAME:
                all_training_infos = file_io.get_trained_adversarial_lense_models(kedp, kecp)
            elif ke_dirname == nc.KNOWLEDGE_ADVERSARIAL_DIRNAME:
                all_training_infos = file_io.get_trained_ke_adv_models(kedp, kecp)
            elif ke_dirname == nc.KNOWLEDGE_UNUSEABLE_DIRNAME:
                all_training_infos = file_io.get_trained_ke_unuseable_models(kedp, kecp)
            else:
                raise NotImplementedError
        if len(all_training_infos) == 0:
            clean_up_after_processing(kedp)
            continue

        # Do the baseline model creation if it not already exists!
        trainer = EvalTrainer(model_infos=all_training_infos)
        trainer.measure_generalization()
        clean_up_after_processing(kedp)
    return


if __name__ == "__main__":
    main()
    sys.exit(0)
