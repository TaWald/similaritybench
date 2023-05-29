from __future__ import annotations

from pathlib import Path

architecture_choices = "VGG16 VGG19 ResNet34 ResNet50 ResNet101 DenseNet121 DenseNet161".split(" ")

#  ----------- Knowledge Extension -----------


class KENameEncoder:
    @staticmethod
    def encode(
        experiment_description: str,
        dataset: str,
        architecture: str,
        hook_positions: list[int],
        transfer_depth: int,
        kernel_width: int,
        group_id: int,
        sim_loss: str,
        sim_loss_weight: float,
        dis_loss: str,
        dis_loss_weight: float,
        ce_loss_weight: float,
        aggregate_reps: int,
        softmax_metrics: int,
        epochs_before_regularization: int,
    ):
        hooks = "-".join([str(h) for h in hook_positions])
        return (
            f"{experiment_description}__{dataset}__{architecture}__GroupID_{group_id}"
            + f"__Hooks_{hooks}__TDepth_{transfer_depth}"
            + f"__KWidth_{kernel_width}__Sim_{sim_loss}_{sim_loss_weight:.02f}"
            + f"__Dis_{dis_loss}_{dis_loss_weight:.02f}_{ce_loss_weight:.02f}"
            + f"__ar_{aggregate_reps}__sm_{softmax_metrics}__ebr_{epochs_before_regularization}"
        )

    @staticmethod
    def decode(
        dirname: str | Path,
    ) -> tuple[str, str, str, list[int], int, int, int, str, float, str, float, float, bool, bool, int]:
        """Decodes the Directory name that has been encoded.
        :returns experiment name, hook_id, transfer_depth, transfer_width, kernel_width
        """

        values = str(dirname).split("__")
        try:
            (
                exp_description,
                dataset,
                architecture,
                group_id,
                hook_positions,
                tdepth,
                kwidth,
                sim_l,
                dis_l,
                aggregate,
                softmax_metrics,
                epochs_before_regularization,
            ) = values
        except ValueError as e:
            raise ValueError(f"{dirname} seems not to be up to current knowledge_extension naming standards.") from e
        group_id_i: int = int(group_id.split("_")[-1])
        joined_hooks: str = hook_positions.split("_")[-1]
        hooks: list[int] = [int(a) for a in joined_hooks.split("-")]
        tdepth_i: int = int(tdepth.split("_")[-1])
        kwidth_i: int = int(kwidth.split("_")[-1])
        sim_stuff: list = sim_l.split("_")
        sim_loss, sim_loss_weight = sim_stuff[1], float(sim_stuff[2])
        dis_stuff: list = dis_l.split("_")
        dis_loss, dis_loss_weight, ce_loss_weight = dis_stuff[1], float(dis_stuff[2]), float(dis_stuff[3])
        agg: bool = bool(aggregate.split("_")[-1])
        sm: bool = bool(softmax_metrics.split("_")[-1])
        ebr: int = int(epochs_before_regularization.split("_")[-1])

        return (
            exp_description,
            dataset,
            architecture,
            hooks,
            tdepth_i,
            kwidth_i,
            group_id_i,
            sim_loss,
            sim_loss_weight,
            dis_loss,
            dis_loss_weight,
            ce_loss_weight,
            agg,
            sm,
            ebr,
        )


class KEUnusableDownstreamNameEncoder:
    @staticmethod
    def encode(
        experiment_description: str,
        dataset: str,
        architecture: str,
        hook_positions: list[int],
        transfer_depth: int,
        kernel_width: int,
        group_id: int,
        transfer_loss_weight: float,
        gradient_reversal_scale: float,
        ce_loss_weight: float,
        epochs_before_regularization: int,
    ):
        hooks = "-".join([str(h) for h in hook_positions])
        return (
            f"{experiment_description}__{dataset}__{architecture}__GroupID_{group_id}"
            + f"__Hooks_{hooks}__TDepth_{transfer_depth}"
            + f"__KWidth_{kernel_width}__Trans_{transfer_loss_weight:.02f}_"
            + f"{gradient_reversal_scale:.02f}_{ce_loss_weight:.02f}"
            + f"__ebr_{epochs_before_regularization}"
        )

    @staticmethod
    def decode(
        dirname: str | Path,
    ) -> tuple[str, str, str, list[int], int, int, int, float, float, float, int]:
        """Decodes the Directory name that has been encoded.
        :returns experiment name, hook_id, transfer_depth, transfer_width, kernel_width
        """

        values = str(dirname).split("__")
        (
            exp_description,
            dataset,
            architecture,
            group_id,
            hook_positions,
            tdepth,
            kwidth,
            trans_l,
            epochs_before_regularization,
        ) = values
        group_id_i: int = int(group_id.split("_")[-1])
        joined_hooks: str = hook_positions.split("_")[-1]
        hooks: list[int] = [int(a) for a in joined_hooks.split("-")]
        tdepth_i: int = int(tdepth.split("_")[-1])
        kwidth_i: int = int(kwidth.split("_")[-1])
        trans_stuff: list = trans_l.split("_")
        trans_loss_weight, grs, ce_loss_weight, = (
            trans_stuff[1],
            float(trans_stuff[2]),
            float(trans_stuff[3]),
        )
        ebr: int = int(epochs_before_regularization.split("_")[-1])

        return (
            exp_description,
            dataset,
            architecture,
            hooks,
            tdepth_i,
            kwidth_i,
            group_id_i,
            trans_loss_weight,
            grs,
            ce_loss_weight,
            ebr,
        )


class KEAdversarialNameEncoder:
    @staticmethod
    def encode(
        experiment_description: str,
        dataset: str,
        architecture: str,
        hook_positions: list[int],
        transfer_depth: int,
        kernel_width: int,
        group_id: int,
        adv_loss: str,
        adv_loss_weight: float,
        grad_rev_scale: float,
        ce_loss_weight: float,
        epochs_before_regularization: int,
    ):
        hooks = "-".join([str(h) for h in hook_positions])
        return (
            f"{experiment_description}__{dataset}__{architecture}__GroupID_{group_id}"
            + f"__Hooks_{hooks}__TDepth_{transfer_depth}"
            + f"__KWidth_{kernel_width}__Adv_{adv_loss}_{adv_loss_weight:.02f}"
            + f"_{grad_rev_scale:.02f}_{ce_loss_weight:.02f}"
            + f"__ebr_{epochs_before_regularization}"
        )

    @staticmethod
    def decode(
        dirname: str | Path,
    ) -> tuple[str, str, str, list[int], int, int, int, str, float, float, float, int]:
        """Decodes the Directory name that has been encoded.
        :returns experiment name, hook_id, transfer_depth, transfer_width, kernel_width
        """

        values = str(dirname).split("__")
        (
            exp_description,
            dataset,
            architecture,
            group_id,
            hook_positions,
            tdepth,
            kwidth,
            adv_l,
            epochs_before_regularization,
        ) = values
        group_id_i: int = int(group_id.split("_")[-1])
        joined_hooks: str = hook_positions.split("_")[-1]
        hooks: list[int] = [int(a) for a in joined_hooks.split("-")]
        tdepth_i: int = int(tdepth.split("_")[-1])
        kwidth_i: int = int(kwidth.split("_")[-1])
        adv_stuff: list = adv_l.split("_")
        adv_loss, adv_loss_weight, adv_grs, ce_loss = (
            adv_stuff[1],
            float(adv_stuff[2]),
            float(adv_stuff[3]),
            float(adv_stuff[4]),
        )
        ebr: int = int(epochs_before_regularization.split("_")[-1])

        return (
            exp_description,
            dataset,
            architecture,
            hooks,
            tdepth_i,
            kwidth_i,
            group_id_i,
            adv_loss,
            adv_loss_weight,
            adv_grs,
            ce_loss,
            ebr,
        )


class KESubNameEncoder:
    @staticmethod
    def encode(
        experiment_description: str,
        dataset: str,
        architecture: str,
        hook_positions: list[int],
        transfer_depth: int,
        kernel_width: int,
        group_id: int,
        sim_loss: str,
        sim_loss_weight: float,
        ce_loss_weight: float,
        epochs_before_regularization: int,
    ):
        hooks = "-".join([str(h) for h in hook_positions])
        return (
            f"{experiment_description}__{dataset}__{architecture}__GroupID_{group_id}"
            + f"__Hooks_{hooks}__TDepth_{transfer_depth}"
            + f"__KWidth_{kernel_width}__Sim_{sim_loss}_{sim_loss_weight:.02f}_{ce_loss_weight:.02f}"
            + f"__ebr_{epochs_before_regularization}"
        )

    @staticmethod
    def decode(
        dirname: str | Path,
    ) -> tuple[str, str, str, list[int], int, int, int, str, float, float, int]:
        """Decodes the Directory name that has been encoded.
        :returns experiment name, hook_id, transfer_depth, transfer_width, kernel_width
        """

        values = str(dirname).split("__")
        (
            exp_description,
            dataset,
            architecture,
            group_id,
            hook_positions,
            tdepth,
            kwidth,
            sim_l,
            epochs_before_regularization,
        ) = values
        group_id_i: int = int(group_id.split("_")[-1])
        joined_hooks: str = hook_positions.split("_")[-1]
        hooks: list[int] = [int(a) for a in joined_hooks.split("-")]
        tdepth_i: int = int(tdepth.split("_")[-1])
        kwidth_i: int = int(kwidth.split("_")[-1])
        sim_stuff: list = sim_l.split("_")
        sim_loss, sim_loss_weight, ce_loss_weight = sim_stuff[1], float(sim_stuff[2]), float(sim_stuff[3])
        ebr: int = int(epochs_before_regularization.split("_")[-1])

        return (
            exp_description,
            dataset,
            architecture,
            hooks,
            tdepth_i,
            kwidth_i,
            group_id_i,
            sim_loss,
            sim_loss_weight,
            ce_loss_weight,
            ebr,
        )


class KEOutputNameEncoder:
    @staticmethod
    def encode(
        experiment_description: str,
        dataset: str,
        architecture: str,
        group_id: int,
        dis_loss: str,
        dis_loss_weight: str,
        ce_loss_weight: float,
        softmax_metrics: int,
        epochs_before_regularization: int,
        pc_grad: int,
    ):
        return (
            f"{experiment_description}__{dataset}__{architecture}__GroupID_{group_id}"
            + f"__Dis_{dis_loss}_{dis_loss_weight}_{ce_loss_weight:.02f}"
            + f"__sm_{softmax_metrics}__ebr_{epochs_before_regularization}__pcg_{pc_grad}"
        )

    @staticmethod
    def decode(
        dirname: str | Path,
    ) -> tuple[str, str, str, int, str, str, float, bool, int, bool]:
        """Decodes the Directory name that has been encoded.
        :returns experiment name, hook_id, transfer_depth, transfer_width, kernel_width
        """

        values = str(dirname).split("__")
        (
            exp_description,
            dataset,
            architecture,
            group_id,
            dis_l,
            softmax_metrics,
            epochs_before_regularization,
            pcg,
        ) = values
        pc_grad: bool = bool(pcg.split("_")[-1])

        group_id_i: int = int(group_id.split("_")[-1])
        dis_stuff: list = dis_l.split("_")
        dis_loss, dis_loss_weight, ce_loss_weight = dis_stuff[1], dis_stuff[2], float(dis_stuff[3])
        sm: bool = bool(softmax_metrics.split("_")[-1])
        ebr: int = int(epochs_before_regularization.split("_")[-1])

        return (
            exp_description,
            dataset,
            architecture,
            group_id_i,
            dis_loss,
            dis_loss_weight,
            ce_loss_weight,
            sm,
            ebr,
            pc_grad,
        )


class PretrainedNameEncoder:
    @staticmethod
    def encode(
        experiment_description: str,
        dataset: str,
        architecture: str,
        group_id: int,
        pretrained: int,
        warmup_pretrained: int,
        linear_probe_only: int,
    ):
        return (
            f"{experiment_description}__{dataset}__{architecture}__GroupID_{group_id}"
            + f"__Pretrained_{pretrained}__WarmUp_{warmup_pretrained}__LinearProbeOnly_{linear_probe_only}"
        )

    @staticmethod
    def decode(
        dirname: str | Path,
    ) -> tuple[str, str, str, int, bool, bool, bool]:
        """Decodes the Directory name that has been encoded.
        :returns experiment name, hook_id, transfer_depth, transfer_width, kernel_width
        """

        values = str(dirname).split("__")
        (
            exp_description,
            dataset,
            architecture,
            group_id,
            pretrained,
            frozen_for,
            lpo,
        ) = values
        group_id_i: int = int(group_id.split("_")[-1])
        was_pretrained: bool = bool(int(pretrained.split("_")[-1]))
        warmup_pretrained: bool = bool(int(frozen_for.split("_")[-1]))
        linear_probe_only: bool = bool(int(lpo.split("_")[-1]))

        return (
            exp_description,
            dataset,
            architecture,
            group_id_i,
            was_pretrained,
            warmup_pretrained,
            linear_probe_only,
        )


class KEAdversarialLenseOutputNameEncoder:
    @staticmethod
    def encode(
        experiment_description: str,
        dataset: str,
        architecture: str,
        group_id: int,
        adv_loss: str,
        lense_adversarial_weight: float,
        ce_loss_weight: float,
        lense_reconstruction_weight: float,
        lense_setting: str,
    ):
        return (
            f"{experiment_description}__{dataset}__{architecture}__GroupID_{group_id}"
            + f"__{lense_setting}__Adv_{adv_loss}_{lense_adversarial_weight:.04f}_"
            + f"{lense_reconstruction_weight:.04f}_{ce_loss_weight:.04f}"
        )

    @staticmethod
    def decode(
        dirname: str | Path,
    ) -> tuple[str, str, str, int, str, str, float, float, float]:
        """Decodes the Directory name that has been encoded.
        :returns experiment name, hook_id, transfer_depth, transfer_width, kernel_width
        """

        values = str(dirname).split("__")
        (
            exp_description,
            dataset,
            architecture,
            group_id,
            lense_setting,
            adv_l,
        ) = values
        group_id_i: int = int(group_id.split("_")[-1])
        adv_stuff: list = adv_l.split("_")
        adv_loss, adv_loss_weight, lense_rc_weight, ce_loss_weight = (
            adv_stuff[1],
            float(adv_stuff[2]),
            float(adv_stuff[3]),
            float(adv_stuff[4]),
        )

        return (
            exp_description,
            dataset,
            architecture,
            group_id_i,
            lense_setting,
            adv_loss,
            adv_loss_weight,
            lense_rc_weight,
            ce_loss_weight,
        )


KNOWLEDGE_EXTENSION_DIRNAME_SCIS = "knowledge_extension_scis"
KNOWLEDGE_UNUSEABLE_DIRNAME = "knowledge_extension_unusable_downstream"
KNOWLEDGE_ADVERSARIAL_DIRNAME = "knowledge_adversarial_extension"
KE_ADVERSARIAL_LENSE_DIRNAME = "knowledge_extension_adversarial_lense"
KE_OUTPUT_REGULARIZATION_DIRNAME = "knowledge_extension_output_reg"
PRETRAINED_TEST_DIRNAME = "test_pretraining"
KE_SUB_DIRNAME = "knowledge_extension_subtraction"

SINGLE_RESULTS_FILE = "single_results.json"
ENSEMBLE_RESULTS_FILE = "ensemble_results.json"
CALIBRATED_ENSEMBLE_RESULTS_FILE = "calibrated_ensemble_results.json"

ROB_SINGLE_FILE = "robustness_single.json"
ROB_ENSEMBLE_FILE = "robustness_ensemble.json"
ROB_CALIBRATED_ENSEMBLE_FILE = "robustness_calibrated_ensemble.json"

KE_LENSE_CKPT_NAME = "lense_checkpoint.ckpt"
KE_INFO_FILE = "info.json"
LENSE_INFO = "lense_info.json"

# ----------------- Logging ------------------
LOG_DIR = "LOGS"

# ----------------- Activation ------------------
TEST_ACTI_TMPLT = "test_activ_{}.npy"
TEST_ACTI_RE = r"^test_activ_((bn)|(conv)|(id))\d+\.npy$"
VAL_ACTI_TMPLT = "val_activ_{}.npy"
VAL_ACTI_RE = r"^val_activ_((bn)|(conv)|(id))\d+\.npy$"
HIGHEST_ACTI_ID_VALS = "channelwise_most_activating_inputs.json"

# ----------------- Prediction ------------------
MODEL_TRAIN_PD_TMPLT = "train_prediction.npy"
MODEL_TEST_PD_TMPLT = "test_prediction.npy"
MODEL_VAL_PD_TMPLT = "val_prediction.npy"
MODEL_TEST_CALIB_PD_TMPLT = "test_prediction_calibrated.npy"

MODEL_TRAIN_GT_TMPLT = "train_groundtruth.npy"
MODEL_TEST_GT_TMPLT = "test_groundtruth.npy"
MODEL_VAL_GT_TMPLT = "val_groundtruth.npy"

MODEL_TRAIN_GT_RE = r"^train_prediction\.npy"
MODEL_TEST_PD_RE = r"^test_prediction\.npy"

MODEL_TEST_KEYVALS = "prediction_keyvals.json"

# ----------------- Prediction ------------------
GNRLZ_PD_TMPLT = "augmented_PD_{}_{}.npy"
GNRLZ_GT_TMPLT = "augmented_GT_{}_{}.npy"
GNRLZ_OUT_RESULTS = "augmented_results.json"
ROBUST_SINGLE_RESULTS = "single_robustness_results.json"
ROBUST_ENSEMBLE_RESULTS = "ensemble_robustness_results.json"
ROBUST_CALIB_ENS_RESULTS = "calib_ensemble_robustness_results.json"

# For the transfer setting: Includes layer indicator to know transfer creating preds

# ----------------- First Models ------------------
KE_FIRST_MODEL_DIR = "FIRST_MODELS__{}__{}"
KE_FIRST_MODEL_GROUP_ID_DIR = "groupid_{}"

PCA_TRANSFORM = "pca_trans_{}.npy"

# Comparison/Similarity
COMP_ARCH_PRUNE_TMPLT = "arch_{}_to_{}"  # _{VGG19-LotteryTicket-95.6}_
COMP_LAYER_TMPLT = "src_{}_tgt_{}.npy"
COMP_DONE_FLAG = "done.txt"
MODEL_SCORE_RES_TMPLT = "layerwise_scores_{}by{}.json"
LAYERWISE_SCORE_RES_TMPLT = "all_scores_by_layer.json"


# ----------------- Fusion/Transfer Result ---------------
FUSION_DIR_TMPLT = "fusion_{}_{}"
FUSION_DIR_RE = r"^fusion_\d+_\d+$"

TRAINED_TRANS_DIR_TMPLT = "transfer_{}_{}_{}"
TRAINED_TRANS_DIR_RE = r"^transfer_\d+_\d+_\d+$"

# ----------------- Noise ---------------
# NOISY STUFF
LAYERWISE_NOISE_SENSITIVITY_TMPLT = "noise_sensitivity_results.json"

# Visualization Key
UP_TO_DATE_TMPLT = "no_overwrite.txt"
# No REGEX needed for that.


GLOBAL_PD_TMPLT = "{}_global_partial_predictions.npy"

MEAN_ORI_APPROX_ACTI_TMPLT = "layerwise_channelwise_mean_activation_of_original_approximation.json"

#
MODEL_NAME_TMPLT = "model_{:04d}"
MODEL_NAME_RE = r"model_\d{4}$"

PRUNED_MODEL_NAME_TMPLT = "pruned_model_{:04d}"
PRUNED_MODEL_NAME_RE = r"pruned_model_\d{4}$"
PRUNED_CKPT = "pruned.ckpt"
PRUNE_RES = "pruning_result.json"
PRUNE_DETAILS = "pruning_result_details.json"
PRUNE_DIR = "pruning"

DISTIL_DIR = "distillation"
DISTIL_MODEL_NAME_TMPLT = "distil_mID_{}_arch_{}_source_mID_{}_arch_{}"

STATIC_CKPT_NAME = "final.ckpt"
APPROX_CKPT_NAME = "approx_layer_{}.ckpt"
APPROX_CKPT_INFO_NAME = "approx_layer_info_{}.json"
CKPT_NAME_TMPLT = "epoch_{}.ckpt"
CKPT_NAME_RE = r"^epoch_\d+.ckpt$"

OUTPUT_TMPLT = "output.json"
LENSE_TMPLT = "lense_output.json"
LAST_METRICS_TMPLT = "last_metrics.json"
CALIB_TMPLT = "calibration.json"
GENERALIZATION_TMPLT = "generalization.json"

LENSE_EXAMPLE_DIR_NAME = "lense_examples"
CKPT_DIR_NAME = "checkpoints"
ACTI_DIR_NAME = "activations"
PRED_DIR_NAME = "predictions"
REL_DIR_NAME = "channel_relevance"
COMP_DIR_NAME = "comparisons"
