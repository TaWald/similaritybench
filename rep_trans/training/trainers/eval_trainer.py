from __future__ import annotations

from pytorch_lightning import Trainer

from augmented_datasets.scripts.get_dataloaders import get_augmented_cifar10_test_dataloader
from rep_trans.training.ke_train_modules.EnsembleEvaluationLightningModule import EnsembleEvaluationLightningModule
from rep_trans.util import data_structs as ds
from rep_trans.util import file_io
from rep_trans.util import name_conventions as nc
from rep_trans.util.gpu_cluster_worker_nodes import get_workers_for_current_node


class EvalTrainer:
    def __init__(
            self,
            model: EnsembleEvaluationLightningModule,
            params: ds.Params,
            arch_params: dict,
            ):
        self.model: EnsembleEvaluationLightningModule = model
        self.params = params
        self.arch_params = arch_params
        self.num_workers = get_workers_for_current_node()
        # Create them. Should not exist though or overwrite would happen!

        self.test_kwargs = {
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "batch_size": self.params.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            }
        
    def measure_generalization(self):
        if self.params.dataset != "CIFAR10":
            raise NotImplementedError("Generalization measurement only works for ")
        trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=None,
            accelerator="gpu",
            devices=1,
            precision=32,
            default_root_dir=None,
            enable_progress_bar=False,
            logger=False,
            profiler=None,
            )
        self.model.load_latest_checkpoint()
        self.model.cuda()
        self.model.eval()
        self.model.clear_outputs = False
        dataloaders = get_augmented_cifar10_test_dataloader(self.test_kwargs)
        n_models = len(self.model.models)
        all_results = [{} for _ in range(n_models)]
        for dl in dataloaders:
            trainer.validate(self.model, dl.dataloader)
            final_metrics = self.model.all_metrics
            for i in range(n_models):
                result = final_metrics[i]
                if dl.name in result[i].keys():
                    all_results[i][dl.name].update({str(dl.value): final_metrics})
                else:
                    all_results[i][dl.name] = {str(dl.value): final_metrics}
        for i in range(n_models):
            file_io.save(all_results[i], self.model.infos[i].path_ckpt_root, nc.GNRLZ_OUT_RESULTS)