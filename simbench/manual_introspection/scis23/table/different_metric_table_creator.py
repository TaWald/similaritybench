from pathlib import Path

import pandas as pd
from simbench.util.file_io import load_json


def average_multiple_metrics(res: list[dict]):
    all_avg_results = []

    loss_weights = ["0.25", "1.00", "4.00"]
    metrics = ["L2Corr", "ExpVar", "LinCKA"]
    layers = ["1", "3", "8", "13"]
    for lw in loss_weights:
        for m in metrics:
            for l in layers:
                filtered_res = [r for r in res if r["Metric"] == m and r["Layer"] == l and r["Loss Weight"] == lw]
                if len(filtered_res) == 0:
                    continue
                tmp = pd.DataFrame(filtered_res)

                avg_result = dict()
                avg_result["N Models"] = 2
                avg_result["Metric"] = filtered_res[0]["Metric"]
                avg_result["Layer"] = filtered_res[0]["Layer"]
                avg_result["Loss Weight"] = filtered_res[0]["Loss Weight"]
                avg_result["Ensemble Acc"] = tmp["ensemble_accuracy"].mean()
                avg_result["Ensemble Acc Std"] = tmp["ensemble_accuracy"].std()
                avg_result["New Model Acc"] = tmp["new_model_accuracy"].mean()
                avg_result["New Model Acc Std"] = tmp["new_model_accuracy"].std()
                avg_result["Cohens Kappa"] = tmp["cohens_kappa"].mean()
                avg_result["Cohens Kappa Std"] = tmp["cohens_kappa"].std()
                avg_result["Rel. Ensemble Acc"] = tmp["relative_ensemble_performance"].mean()
                avg_result["Rel. Ensemble Acc Std"] = tmp["relative_ensemble_performance"].std()
                avg_result["ERD"] = tmp["error_ratio"].mean()
                avg_result["ERD Std"] = tmp["error_ratio"].std()
                avg_result["JSD"] = tmp["jensen_shannon_div"].mean()
                avg_result["JSD Std"] = tmp["jensen_shannon_div"].std()

                all_avg_results.append(avg_result)
    return all_avg_results


def average_baselines(res: list[dict]):
    all_avg_results = []

    tmp = pd.DataFrame(res)

    avg_result = dict()
    avg_result["N Models"] = 2
    avg_result["Metric"] = "None"
    avg_result["Layer"] = "None"
    avg_result["Loss Weight"] = "0.00"
    avg_result["Ensemble Acc"] = tmp["ensemble_accuracy"].mean()
    avg_result["Ensemble Acc Std"] = tmp["ensemble_accuracy"].std()
    avg_result["New Model Acc"] = tmp["new_model_accuracy"].mean()
    avg_result["New Model Acc Std"] = tmp["new_model_accuracy"].std()
    avg_result["Cohens Kappa"] = tmp["cohens_kappa"].mean()
    avg_result["Cohens Kappa Std"] = tmp["cohens_kappa"].std()
    avg_result["Rel. Ensemble Acc"] = tmp["relative_ensemble_performance"].mean()
    avg_result["Rel. Ensemble Acc Std"] = tmp["relative_ensemble_performance"].std()
    avg_result["ERD"] = tmp["error_ratio"].mean()
    avg_result["ERD Std"] = tmp["error_ratio"].std()
    avg_result["JSD"] = tmp["jensen_shannon_div"].mean()
    avg_result["JSD Std"] = tmp["jensen_shannon_div"].std()

    all_avg_results.append(avg_result)
    return all_avg_results


def main():
    json_dir = Path("/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/scis23/representation_comp_results")
    # baseline_json = ["functional_scis23_baseline_ensemble_first_two.json"]
    jsons_of_interest: list[str] = [
        "functional_diff_metrics_layer_1_tdepth_1_ExpVar_0.25.json",
        "functional_diff_metrics_layer_1_tdepth_1_ExpVar_1.00.json",
        "functional_diff_metrics_layer_1_tdepth_1_ExpVar_4.00.json",
        "functional_diff_metrics_layer_1_tdepth_1_L2Corr_0.25.json",
        "functional_diff_metrics_layer_1_tdepth_1_L2Corr_1.00.json",
        "functional_diff_metrics_layer_1_tdepth_1_L2Corr_4.00.json",
        "functional_diff_metrics_layer_1_tdepth_1_LinCKA_0.25.json",
        "functional_diff_metrics_layer_1_tdepth_1_LinCKA_1.00.json",
        "functional_diff_metrics_layer_1_tdepth_1_LinCKA_4.00.json",
        "functional_diff_metrics_layer_3_tdepth_1_ExpVar_0.25.json",
        "functional_diff_metrics_layer_3_tdepth_1_ExpVar_1.00.json",
        "functional_diff_metrics_layer_3_tdepth_1_ExpVar_4.00.json",
        "functional_diff_metrics_layer_3_tdepth_1_L2Corr_0.25.json",
        "functional_diff_metrics_layer_3_tdepth_1_L2Corr_1.00.json",
        "functional_diff_metrics_layer_3_tdepth_1_L2Corr_4.00.json",
        "functional_diff_metrics_layer_3_tdepth_1_LinCKA_0.25.json",
        "functional_diff_metrics_layer_3_tdepth_1_LinCKA_1.00.json",
        "functional_diff_metrics_layer_3_tdepth_1_LinCKA_4.00.json",
        "functional_diff_metrics_layer_8_tdepth_1_ExpVar_0.25.json",
        "functional_diff_metrics_layer_8_tdepth_1_ExpVar_1.00.json",
        "functional_diff_metrics_layer_8_tdepth_1_ExpVar_4.00.json",
        "functional_diff_metrics_layer_8_tdepth_1_L2Corr_0.25.json",
        "functional_diff_metrics_layer_8_tdepth_1_L2Corr_1.00.json",
        "functional_diff_metrics_layer_8_tdepth_1_L2Corr_4.00.json",
        "functional_diff_metrics_layer_8_tdepth_1_LinCKA_0.25.json",
        "functional_diff_metrics_layer_8_tdepth_1_LinCKA_1.00.json",
        "functional_diff_metrics_layer_8_tdepth_1_LinCKA_4.00.json",
        "functional_diff_metrics_layer_13_tdepth_1_ExpVar_0.25.json",
        "functional_diff_metrics_layer_13_tdepth_1_ExpVar_1.00.json",
        "functional_diff_metrics_layer_13_tdepth_1_ExpVar_4.00.json",
        "functional_diff_metrics_layer_13_tdepth_1_L2Corr_0.25.json",
        "functional_diff_metrics_layer_13_tdepth_1_L2Corr_1.00.json",
        "functional_diff_metrics_layer_13_tdepth_1_L2Corr_4.00.json",
        "functional_diff_metrics_layer_13_tdepth_1_LinCKA_0.25.json",
        "functional_diff_metrics_layer_13_tdepth_1_LinCKA_1.00.json",
        "functional_diff_metrics_layer_13_tdepth_1_LinCKA_4.00.json",
    ]
    all_results: list[pd.DataFrame] = []
    for json_of_interest in jsons_of_interest:
        res = load_json(json_dir / json_of_interest)
        metric = json_of_interest.split("_")[-2]
        layer = json_of_interest.split("_")[4]
        loss_weight = json_of_interest.split("_")[-1].split(".json")[0]
        [r.update({"Metric": metric, "Layer": layer, "Loss Weight": loss_weight}) for r in res]
        avg_simple_res = average_multiple_metrics(res)
        if len(avg_simple_res) == 0:
            continue
        all_results.append(pd.DataFrame(avg_simple_res))

    # Grab baseline:
    # baseline_res = load_json(json_dir / baseline_json[0])
    # baseline = [average_baselines(baseline_res)]

    all_results_df = pd.concat(all_results)

    exp_var_loss_effect = all_results_df[all_results_df["Metric"] == "ExpVar"]
    lincka_loss_effect = all_results_df[all_results_df["Metric"] == "LinCKA"]
    l2corr_loss_effect = all_results_df[all_results_df["Metric"] == "L2Corr"]

    joined_df = pd.concat(all_results)
    joined_df = joined_df.set_index("N Models").T
    exp_var_loss_effect.to_latex(
        "metrics_and_lossweights_effects_cifar10_resnet34_expvar.tex",
        index=False,
        float_format="{:0.3%}".format,
        columns=[
            "Metric",
            "Layer",
            "Loss Weight",
            "Ensemble Acc",
            "Ensemble Acc Std",
            "New Model Acc",
            "New Model Acc Std",
            "Cohens Kappa",
            "Cohens Kappa Std",
            "JSD",
            "JSD Std",
            "ERD",
            "ERD Std",
        ],
    )
    lincka_loss_effect.to_latex(
        "metrics_and_lossweights_effects_cifar10_resnet34_lincka.tex",
        index=False,
        float_format="{:0.3%}".format,
        columns=[
            "Metric",
            "Layer",
            "Loss Weight",
            "Ensemble Acc",
            "Ensemble Acc Std",
            "New Model Acc",
            "New Model Acc Std",
            "Cohens Kappa",
            "Cohens Kappa Std",
            "JSD",
            "JSD Std",
            "ERD",
            "ERD Std",
        ],
    )

    l2corr_loss_effect.to_latex(
        "metrics_and_lossweights_effects_cifar10_resnet34_l2corr.tex",
        index=False,
        float_format="{:0.3%}".format,
        columns=[
            "Metric",
            "Layer",
            "Loss Weight",
            "Ensemble Acc",
            "Ensemble Acc Std",
            "New Model Acc",
            "New Model Acc Std",
            "Cohens Kappa",
            "Cohens Kappa Std",
            "JSD",
            "JSD Std",
            "ERD",
            "ERD Std",
        ],
    )

    print(0)


if __name__ == "__main__":
    main()
