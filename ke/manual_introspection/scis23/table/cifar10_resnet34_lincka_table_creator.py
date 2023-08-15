from pathlib import Path

import pandas as pd
from ke.manual_introspection.scis23.table.cifar10_resnet34_expvar_table_creator import average_multiple_ensemble
from ke.util.file_io import load_json


def main():
    json_dir = Path("/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/scis23/representation_comp_results")
    jsons_of_interest = [
        "functional_scis23_baseline_ensemble.json",
        "functional_lincka_5_models_layer_1.json",
        "functional_lincka_5_models_layer_3.json",
        "functional_lincka_5_models_layer_8.json",
        "functional_lincka_5_models_layer_13.json",
    ]
    additional_info = [
        {
            "Metric": "Baseline",
            "Layer": "None",
        },
        {
            "Metric": "LinCKA",
            "Layer": "1",
        },
        {
            "Metric": "LinCKA",
            "Layer": "3",
        },
        {
            "Metric": "LinCKA",
            "Layer": "8",
        },
        {
            "Metric": "LinCKA",
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
    joined_df = joined_df.set_index("N Models").T
    joined_df.to_latex(
        "lincka_R34_C10_ensemble_results.tex",
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


if __name__ == "__main__":
    main()
