from rep_trans.losses.output_losses.adaptive_diversity_promoting_regularization import AdaptiveDiversityPromotion
from rep_trans.losses.output_losses.ensemble_entropy_maximization import EnsembleEntropyMaximization
from rep_trans.losses.output_losses.entropy_weighted_boosting import EntropyWeightedBoosting
from rep_trans.losses.output_losses.false_predicted_crossentropy import FalsePredictedCrossEntropy
from rep_trans.losses.output_losses.focal_ensemble_entropy_maximization import FocalEnsembleEntropyMaximization
from rep_trans.losses.output_losses.focal_orthogonal_loss import FocalCosineSimProbability
from rep_trans.losses.output_losses.negative_class_crossentropy import NegativeClassCrossEntropy
from rep_trans.losses.output_losses.orthogonal_probability_false_classes import OrthogonalProbabilityNegativeClasses
from rep_trans.losses.output_losses.orthogonality import OrthogonalLogitLoss
from rep_trans.losses.output_losses.pseudo_knowledge_extension import PseudoKnowledgeExtension1
from rep_trans.losses.output_losses.pseudo_knowledge_extension import PseudoKnowledgeExtension2
from rep_trans.losses.output_losses.pseudo_knowledge_extension import PseudoKnowledgeExtension4
from rep_trans.losses.output_losses.pseudo_knowledge_extension import PseudoKnowledgeExtension5
from rep_trans.losses.output_losses.pseudo_knowledge_extension import PseudoKnowledgeExtension8
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from rep_trans.losses.representation_similarity_losses.ke_cka_loss import CKALoss
from rep_trans.losses.representation_similarity_losses.ke_corr import L1CorrLoss
from rep_trans.losses.representation_similarity_losses.ke_corr import L2CorrLoss
from rep_trans.losses.representation_similarity_losses.ke_corr_weighted import WeightedL1CorrLoss
from rep_trans.losses.representation_similarity_losses.ke_exp_var import ExpVarLoss
from rep_trans.losses.representation_similarity_losses.ke_log_corr import LogCorrLoss
from rep_trans.losses.representation_similarity_losses.ke_none_loss import NoneLoss
from rep_trans.losses.representation_similarity_losses.ke_none_loss import NoneOutputLoss
from rep_trans.losses.representation_similarity_losses.ke_relative_reps import EuclideanRelativeRepresentationLoss
from rep_trans.losses.representation_similarity_losses.ke_topk_corr import TopKL2CorrLoss
from rep_trans.losses.representation_similarity_losses.ke_topk_exp_var import TopKExpVarLoss


def find_ke_loss(loss_name: str, softmax_channel_metrics: bool, celu_alpha=3.0) -> AbstractRepresentationLoss:
    if loss_name == "ExpVar":
        return ExpVarLoss(softmax_channel_metrics, celu_alpha)
    elif loss_name == "L2Corr":
        return L2CorrLoss(softmax_channel_metrics)
    elif loss_name == "L1Corr":
        return L1CorrLoss(softmax_channel_metrics)
    elif loss_name == "LogCorr":
        return LogCorrLoss(softmax_channel_metrics)
    elif loss_name == "LinCKA":
        return CKALoss(softmax_channel_metrics)
    elif loss_name == "None":
        return NoneLoss(softmax_channel_metrics)
    elif loss_name == "WL1Corr":
        return WeightedL1CorrLoss(softmax_channel_metrics)
    elif loss_name == "TopkL2Corr":
        return TopKL2CorrLoss(softmax_channel_metrics)
    elif loss_name == "TopkExpVar":
        return TopKExpVarLoss(softmax_channel_metrics)
    elif loss_name == "L2RelRep":
        return EuclideanRelativeRepresentationLoss(softmax_channel_metrics)
    else:
        raise NotImplementedError(f"Unknown loss passed: Got: {loss_name}")


def find_output_ke_loss(loss_name: str, n_classes: int, *args):
    if loss_name == "NegativeClassCrossEntropy":
        return NegativeClassCrossEntropy(n_classes)
    elif loss_name == "FalsePredictedCrossEntropy":
        return FalsePredictedCrossEntropy(n_classes)
    elif loss_name == "OrthogonalLogit":
        return OrthogonalLogitLoss(n_classes)
    elif loss_name == "OrthogonalProbabilityNegativeClasses":
        return OrthogonalProbabilityNegativeClasses(n_classes)
    elif loss_name == "EnsembleEntropyMaximization":
        return EnsembleEntropyMaximization(n_classes)
    elif loss_name == "EntropyWeightedBoosting":
        return EntropyWeightedBoosting(n_classes)
    elif loss_name == "PseudoKnowledgeExtension1":
        return PseudoKnowledgeExtension1(n_classes)
    elif loss_name == "PseudoKnowledgeExtension1":
        return PseudoKnowledgeExtension1(n_classes)
    elif loss_name == "PseudoKnowledgeExtension2":
        return PseudoKnowledgeExtension2(n_classes)
    elif loss_name == "PseudoKnowledgeExtension4":
        return PseudoKnowledgeExtension4(n_classes)
    elif loss_name == "PseudoKnowledgeExtension5":
        return PseudoKnowledgeExtension5(n_classes)
    elif loss_name == "PseudoKnowledgeExtension8":
        return PseudoKnowledgeExtension8(n_classes)
    elif loss_name == "AdaptiveDiversityPromotion":
        return AdaptiveDiversityPromotion(n_classes, *args)
    elif loss_name == "FocalCosineSimProbability":
        return FocalCosineSimProbability(n_classes, *args)
    elif loss_name == "FocalEnsembleEntropyMaximization":
        return FocalEnsembleEntropyMaximization(n_classes, *args)
    elif loss_name == "None":
        return NoneOutputLoss(n_classes)
    else:
        raise NotImplementedError
