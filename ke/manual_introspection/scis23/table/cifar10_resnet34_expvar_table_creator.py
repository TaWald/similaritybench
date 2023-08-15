from pathlib import Path

import pandas as pd
from ke.util.file_io import load_json


def average_multiple_ensemble(res: list[dict]):
    all_avg_results = []
    for i in range(2, 6):
        models_of_same_number = []
        for r in res:
            if r["n_models"] == i:
                models_of_same_number.append(r)
        tmp = pd.DataFrame(models_of_same_number)
        avg_result = dict()
        avg_result["N Models"] = i
        avg_result["Metric"] = models_of_same_number[0]["Metric"]
        avg_result["Layer"] = models_of_same_number[0]["Layer"]
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
    jsons_of_interest = [
        "functional_scis23_baseline_ensemble.json",
        "functional_expvar_5_models_layer_1.json",
        "functional_expvar_5_models_layer_3.json",
        "functional_expvar_5_models_layer_8.json",
        "functional_expvar_5_models_layer_13.json",
    ]
    additional_info = [
        {
            "Metric": "Baseline",
            "Layer": "None",
        },
        {
            "Metric": "ExpVar",
            "Layer": "1",
        },
        {
            "Metric": "ExpVar",
            "Layer": "3",
        },
        {
            "Metric": "ExpVar",
            "Layer": "8",
        },
        {
            "Metric": "ExpVar",
            "Layer": "13",
        },
    ]
    all_results = []
    for json_of_interest, add_info in zip(jsons_of_interest, additional_info):
        res = load_json(json_dir / json_of_interest)
        [r.update(add_info) for r in res]
        avg_simple_res = average_multiple_ensemble(res)
        all_results.append(pd.DataFrame(avg_simple_res))
    joined_df = pd.concat(all_results)
    joined_df.to_latex(
        "expvar_R34_C10_ensemble_results.tex",
        index=False,
        float_format="{:0.3%}".format,
        columns=[
            "Metric",
            "N Models",
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


if __name__ == "__main__":
    main()
