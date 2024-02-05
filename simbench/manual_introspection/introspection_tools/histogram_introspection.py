from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from simbench.manual_introspection.utils.activation_results import ActivationResult


def create_dataframe_from_np_arrays(results: dict[str : np.ndarray], meta_info: dict = None) -> pd.DataFrame:
    data = []
    keys: list[str] = list(results.keys())
    vals: list[np.ndarray] = [np.reshape(v, -1) for v in list(results.values())]

    np_values = np.stack(vals, axis=-1)
    d = pd.DataFrame(data=np_values, columns=keys)
    if meta_info is not None:
        for mi_k, mi_v in meta_info.items():
            d[mi_k] = mi_v
            data.append(d)
    return pd.concat(data)


def create_dataframe(results: list[ActivationResult], origin: str) -> pd.DataFrame:
    data = []
    for res in results:
        if res.error is None:
            d = pd.DataFrame(data=np.reshape(res.values, -1), columns=["true_values"])
        else:
            d = pd.DataFrame(
                data=np.stack([np.reshape(res.values, -1), np.reshape(res.error, -1)], axis=1),
                columns=["true_values", "error"],
            )
        d["layer"] = res.layer
        d["samples"] = res.samples
        d["origin"] = origin
        data.append(d)
    return pd.concat(data)


def interactive_histogram(aor: np.ndarray, aoi: np.ndarray):
    """Creates an interactive plot of two activations."""

    flat_aor = np.reshape(aor, -1)
    flat_aoi = np.reshape(aoi, -1)

    fig = ff.create_distplot([flat_aor, flat_aoi], ["reference", "interest"], bin_size=0.01)
    fig.show()
    return


def all_layer_interactive_histogram(aor: list[ActivationResult], aoi: list[ActivationResult]):
    ref_df = create_dataframe(aor, "reference")
    aoi_df = create_dataframe(aoi, "interest")

    joined_df = pd.concat([ref_df, aoi_df])

    fig = px.histogram(joined_df, x="true_values", color="layer", pattern_shape="origin", nbins=2000, range_x=(-3, 3))
    fig.show()
    return


def histogram(
    result_df: pd.DataFrame, filter_values: dict[str : list[str | int | float]] | None, output_path: str, title: str
):
    tmp_df = result_df
    if filter_values is not None:
        for k, v in filter_values:
            tmp_df = tmp_df[tmp_df[k].isin(v)]

    fig = px.histogram(
        tmp_df, x="true_values", color="layer", pattern_shape="regularization", nbins=2000, title=title
    )
    fig.write_html(output_path)
    fig.show()
    return
