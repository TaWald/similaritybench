from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from manual_introspection.introspection_tools.histogram_introspection import create_dataframe
from manual_introspection.utils.activation_results import ActivationResult


def all_layer_scatterplot_histogram(moi_with_errors: list[ActivationResult], title: str, output_path: Path):

    ref_df = create_dataframe(moi_with_errors, "original")

    ref_df["layer"] = ref_df["layer"].astype("category")
    fig: go.Figure
    fig = px.scatter(ref_df, x="true_values", y="error", color="layer", title=title)

    fig.write_html(output_path)
    fig.show()
    return


def true_approximation_scatterplot(error_df: pd.DataFrame, title: str, output_path: Path):

    error_df["layer"] = error_df["layer"].astype("category")

    fig: go.Figure
    fig = px.scatter(
        error_df,
        x="true_values",
        y="approximated_values",
        symbol="layer",
        color="regularization",
        title=title,
        trendline="ols",
    )
    fig.write_html(output_path)
    return


def scatterplot(error_df: pd.DataFrame, title: str, output_path: Path):

    error_df["layer"] = error_df["layer"].astype("category")

    fig: go.Figure
    fig = px.scatter(error_df, x="true_values", y="error", symbol="layer", color="regularization", title=title)

    fig.write_html(output_path)
    fig.show()
    return
