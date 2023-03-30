import argparse
import sys
from pathlib import Path

from ke.training.trainers.eval_trainer import EvalTrainer
from ke.util import default_parser_args as dpa
from ke.util import file_io
from ke.util import name_conventions as nc
from ke.util.file_io import get_corresponding_first_model
from scripts.post_train_additions.utils import clean_up_after_processing
from scripts.post_train_additions.utils import should_process_a_dir


def evaluate_sequence(data_path: Path, ckpt_path: Path, ke_dirname: str):
    dir_name = data_path.name
    kedp = data_path
    kecp = ckpt_path
    if dir_name.startswith(nc.KE_FIRST_MODEL_DIR.split("__")[0]):
        return
        # Makes no sense to evaluate a ensemble with a single model!
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
        return

    all_training_infos += [get_corresponding_first_model(all_training_infos[0])]
    all_training_infos = list(sorted(all_training_infos, key=lambda x: x.model_id))

    if not all([inf.model_converged() for inf in all_training_infos]):
        print(f"Not all models converged! Skipping dir: {dir_name}")
        return

    if not all([inf.is_calibrated() for inf in all_training_infos]):
        print(f"Not all models are calibrated! Skipping dir: {dir_name}")
        return

    if not should_process_a_dir(kedp):
        return

    # Do the baseline model creation if it not already exists!
    trainer = EvalTrainer(model_infos=all_training_infos)
    trainer.measure_performance(True, True, True)
    trainer.measure_robustness(True, True, True)
    clean_up_after_processing(kedp)


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
        kedp = ke_data_path / dir_name
        kecp = ke_ckpt_path / dir_name
        evaluate_sequence(kedp, kecp, ke_dirname)
    return


if __name__ == "__main__":
    main()
    sys.exit(0)
