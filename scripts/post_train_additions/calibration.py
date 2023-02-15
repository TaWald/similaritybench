import argparse
import sys
from pathlib import Path

from rep_trans.training.ke_train_modules.EnsembleEvaluationLightningModule import EnsembleEvaluationLightningModule
from rep_trans.training.trainers.eval_trainer import EvalTrainer
from rep_trans.util import data_structs as ds
from rep_trans.util import default_parser_args as dpa
from rep_trans.util import file_io
from rep_trans.util import name_conventions as nc
from rep_trans.util.default_params import get_default_arch_params, get_default_parameters
from rep_trans.util.status_check import is_calibrated
from scripts.post_train_additions.utils import should_process_a_file
from rep_trans.util import status_check as sc


def main():
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.dir_parser_arguments(parser)
    args = parser.parse_args()
    ke_dirname = args.ke_dir_name
    
    if ke_dirname == nc.KNOWLEDGE_EXTENSION_DIRNAME:
        encoder = nc.KENameEncoder
    elif ke_dirname == nc.KE_OUTPUT_REGULARIZATION_DIRNAME:
        encoder = nc.KEOutputNameEncoder
    elif ke_dirname == nc.KE_ADVERSARIAL_LENSE_DIRNAME:
        encoder = nc.KEAdversarialLenseOutputNameEncoder
    elif ke_dirname == nc.KNOWLEDGE_ADVERSARIAL_DIRNAME:
        encoder = nc.KEAdversarialNameEncoder
    elif ke_dirname == nc.KNOWLEDGE_UNUSEABLE_DIRNAME:
        encoder = nc.KEUnusableDownstreamNameEncoder
    else:
        raise NotImplementedError
    
    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())
    
    ke_data_path = base_data_path / ke_dirname
    ke_ckpt_path = base_ckpt_path / ke_dirname
    
    for res in ke_data_path.iterdir():
        dir_name = res.name
        
        kedp = ke_data_path / dir_name
        kecp = ke_ckpt_path / dir_name
        
        if not is_calibrated(kedp):
            pass
        
        if not should_process_a_file(kedp):
            continue
        
        decoded = encoder.decode(dir_name)
        exp_name = decoded[0]
        dataset_name = decoded[1]
        architecture_name = decoded[2]
        prev_training_infos: list[ds.BasicTrainingInfo]
        if ke_dirname == nc.KNOWLEDGE_EXTENSION_DIRNAME:
            group_id = decoded[6]
            prev_training_infos = file_io.get_trained_ke_models(kedp, kecp)
        elif ke_dirname == nc.KE_OUTPUT_REGULARIZATION_DIRNAME:
            group_id = decoded[3]
            prev_training_infos = file_io.get_trained_keo_models(kedp, kecp)
        elif ke_dirname == nc.KE_ADVERSARIAL_LENSE_DIRNAME:
            group_id = decoded[3]
            prev_training_infos = file_io.get_trained_adversarial_lense_models(kedp, kecp)
        elif ke_dirname == nc.KNOWLEDGE_ADVERSARIAL_DIRNAME:
            group_id = decoded[6]
            prev_training_infos = file_io.get_trained_ke_adv_models(kedp, kecp)
        elif ke_dirname == nc.KNOWLEDGE_UNUSEABLE_DIRNAME:
            group_id = decoded[6]
            prev_training_infos = file_io.get_trained_ke_unuseable_models(kedp, kecp)
        else:
            raise NotImplementedError
        
        # Do the baseline model creation if it not already exists!
        p: ds.Params = get_default_parameters(architecture_name, ds.Dataset(dataset_name))
        arch_params = get_default_arch_params(ds.Dataset(dataset_name))
        
        first_model = file_io.get_first_model(
            experiment_description=exp_name,
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
        
        all_training_infos: list[ds.BasicTrainingInfo]
        all_training_infos = [first_model] + prev_training_infos
        
        ensemble_module = EnsembleEvaluationLightningModule(all_training_infos, architecture_name, dataset_name)
        trainer = EvalTrainer(model=ensemble_module, params=p, arch_params=arch_params)
        
        trainer.measure_generalization()
        return


if __name__ == "__main__":
    main()
    sys.exit(0)
