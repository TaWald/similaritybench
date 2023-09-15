from pathlib import Path

import pandas as pd
from ke.manual_introspection.scis23.table.cifar10_resnet34_expvar_table_creator import average_multiple_ensemble
from ke.util.file_io import load_json


def metric_and_lambda_ablations_to_table(
    dis_loss_type: str,
    layers: list[int],
    dis_loss_weights: list[float],
    architecture_name: str,
    dataset_json_name: str,
):
    """
    Create the latex output table for the selected architecture, dataset, dis_loss_type, layers, loww_weights,

    :param dis_loss_type: The dis_loss_name to look for in the json file names Example: 'ExpVar'
    :param layers: The hook positions to look for in the json file names: Example: [1, 3, 8, 13]
    :param loss_weights: The loss weights to look for in the json file names: Example: [0.25, 1.0, 4.0]
    :param architecture_name: The architecture name to look for in the json file names: Example: 'ResNet34'

    """
    json_dir = Path("/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/iclr24/json_results")
    jsons_of_interest = [
        f"functional_two_models__{dataset_json_name}__{architecture_name}__None_0.00__tp_1.json"
    ]  # This should always be the baseline config
    additional_infos = [
        {
            "Metric": "Baseline",
            "Layer": "None",
        }
    ]
    for l in layers:
        for dlw in dis_loss_weights:
            jsons_of_interest.append(
                f"functional_two_models__{dataset_json_name}__{architecture_name}__{dis_loss_type}_{dlw:02f}__tp_{l}.json"
            )
            additional_infos.append({"Metric": dis_loss_type, "Layer": f"{l}", "Loss Weight": f"{dlw:.02f}"})

    all_results = []
    for json_of_interest, add_info in zip(jsons_of_interest, additional_infos):
        res = load_json(json_dir / json_of_interest)
        [r.update(add_info) for r in res]
        avg_simple_res = average_multiple_ensemble(res)
        all_results.append(pd.DataFrame(avg_simple_res))
    joined_df = pd.concat(all_results)
    joined_df = joined_df.set_index("N Models").T
    joined_df.to_latex(
        f"two_models_{dis_loss_type}_{architecture_name}_{dataset_json_name}.tex",
        index=False,
        float_format="{:0.3%}".format,
        columns=[
            "Metric",
            "Layer",
            "Ensemble Acc",
            "Ensemble Acc Std",
            "New Model Acc",
            "New Model Acc Std",
            "Cohens Kappa",
            "Cohens Kappa Std",
            "Rel. Ensemble Acc",
            "Rel. Ensemble Acc Std",
            "JSD",
            "JSD Std",
            "ERD",
            "ERD Std",
        ],
    )
    print(0)


def n_ensembles_to_table(
    dis_loss_type: str,
    layers: list[int],
    dis_loss_weights: list[float],
    architecture_name: str,
    dataset_json_name: str,
):
    """
    Create the latex output table for the selected architecture, dataset, dis_loss_type, layers, loww_weights,

    :param dis_loss_type: The dis_loss_name to look for in the json file names Example: 'ExpVar'
    :param layers: The hook positions to look for in the json file names: Example: [1, 3, 8, 13]
    :param loss_weights: The loss weights to look for in the json file names: Example: [0.25, 1.0, 4.0]
    :param architecture_name: The architecture name to look for in the json file names: Example: 'ResNet34'

    """
    json_dir = Path("/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/iclr24/json_results")
    jsons_of_interest = [
        f"functional_ensemble__{dataset_json_name}__{architecture_name}__baseline.json"
    ]  # This should always be the baseline config
    additional_infos = [
        {
            "Metric": "Baseline",
            "Layer": "None",
            "Loss Weight": 0.00,
        }
    ]
    for l in layers:
        for dlw in dis_loss_weights:
            jsons_of_interest.append(
                f"functional_ensemble__{dataset_json_name}__{architecture_name}__{dis_loss_type}_{dlw:.02f}__tp_{l}.json"
            )
            additional_infos.append({"Metric": dis_loss_type, "Layer": f"{l}", "Loss Weight": f"{dlw:.02f}"})

    all_results = []
    for json_of_interest, add_info in zip(jsons_of_interest, additional_infos):
        res = load_json(json_dir / json_of_interest)
        [r.update(add_info) for r in res]
        avg_simple_res = average_multiple_ensemble(res)
        all_results.append(pd.DataFrame(avg_simple_res))
    joined_df = pd.concat(all_results)

    for dlw in [1.0]:
        tmp_df = joined_df[(joined_df["Loss Weight"] == f"{dlw:.02f}") | (joined_df["Loss Weight"] == 0.00)]
        # Now move the layers to the columns
        ensemble_acc_df = tmp_df.pivot(index=["Layer"], columns="N Models", values=["Ensemble Acc"])
        ensemble_acc_df.columns = ensemble_acc_df.columns.droplevel()
        new_model_acc_df = tmp_df.pivot(index=["Layer"], columns="N Models", values=["New Model Acc"])
        new_model_acc_df.columns = new_model_acc_df.columns.droplevel()
        cohens_kappa_df = tmp_df.pivot(index=["Layer"], columns="N Models", values=["Cohens Kappa"])
        cohens_kappa_df.columns = cohens_kappa_df.columns.droplevel()
        result_df = pd.concat([ensemble_acc_df, new_model_acc_df, cohens_kappa_df])

        result_df.to_latex(
            (Path(__file__).parent)
            / "paper_ready_results"
            / f"n_ensemble_{dis_loss_type}_{dlw:.02f}_{architecture_name}_{dataset_json_name}.tex",
            index=True,
            float_format="{:0.3%}".format,
        )
    print(0)


if __name__ == "__main__":
    n_ensembles_to_table("ExpVar", [1, 3, 8, 13], [0.25, 1.0, 4.0], "ResNet34", "cifar100")
    # metric_and_lambda_ablations_to_table("ExpVar", [1, 3, 8, 13], [0.25, 1.0, 4.0], "ResNet34", "CIFAR10")
