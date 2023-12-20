import re
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import scipy.stats as stats
from matplotlib.axes import Axes


def list_data_files(logdir: Path, method: str) -> List[Path]:
    """
    Given a map name, the specification, and the rewarding method, find the list of
    files where the evaluation data is stored. This list consists of all evaluations
    made across training iterations and random seeds. The caller should infer the
    training iteration from the file name.

    The output will be sorted by file name, i.e., by the iteration number.
    """
    logdir = Path(logdir)
    datadir = logdir / method
    assert datadir.is_dir()
    csv_file_re = re.compile(r"progress\d{5}.csv")

    files = deque()
    for path in datadir.iterdir():
        if path.is_file() and csv_file_re.fullmatch(path.name):
            files.append(path)

    return list(sorted(files, key=lambda p: p.name))


def extract_data(data_files: List[Path]) -> pl.DataFrame:
    """From the list of CSV eval files, extract a dataframe"""

    data = [
        pl.read_csv(f)
        .select(["time/episodes", "eval/success_rate"])
        .filter(pl.col("eval/success_rate").is_not_null())
        for f in data_files
    ]

    return pl.concat(data, how="vertical")


def load_data(logdir: Path, method: str) -> pl.DataFrame:
    return extract_data(list_data_files(logdir, method))


def _get_wilson_center(data: pl.DataFrame, confidence=0.95) -> pl.DataFrame:
    """Given a data of Bernoulli trials, get the Wilson center, the new count, and stddev for the data"""
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - (alpha / 2))

    comp = data.lazy()

    center = comp.groupby("time/episodes", maintain_order=True).agg(
        [
            (pl.sum("eval/success_rate") + (z**2) / 2).alias("sum"),
            (pl.count() + (z**2) / 2).alias("count"),
        ]
    )
    p = center.select(
        [
            pl.col("time/episodes"),
            (pl.col("sum") / pl.col("count")).alias("center"),
            pl.col("count"),
        ]
    )
    ret = p.select(
        [
            pl.col("time/episodes"),
            pl.col("center"),
            (pl.col("center") * (pl.lit(1) - pl.col("center"))).sqrt().alias("std"),
            pl.col("count"),
        ]
    ).sort("time/episodes")
    return ret.collect()


def plot_curve(
    name: str, ax: Axes, data: pl.DataFrame, *, span=5, confidence=0.97, bernoulli=False
):
    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - (alpha / 2))

    aggregates = (
        data.lazy()
        .groupby("time/episodes", maintain_order=True)
        .agg(
            [
                pl.mean("eval/success_rate").alias("mean"),
                pl.std("eval/success_rate").alias("std"),
                pl.count().alias("count"),
            ]
        )
    )

    if bernoulli:
        # Use Agresti–Coull interval
        # Essentially we need to adjust the mean and std deviations first
        wilson_center = _get_wilson_center(data, confidence=confidence).lazy()
        margin = wilson_center.select(
            [
                pl.col("center").alias("mean"),
                (pl.lit(z_critical) * (pl.col("std") / pl.col("count").sqrt())).alias(
                    "margin"
                ),
            ]
        )
    else:
        margin = aggregates.select(
            [
                pl.col("mean"),
                (pl.lit(z_critical) * (pl.col("std") / pl.col("count").sqrt())).alias(
                    "margin"
                ),
            ]
        )
    interval = margin.select(
        [
            (pl.col("mean") - pl.col("margin")).alias("low"),
            (pl.col("mean") + pl.col("margin")).alias("high"),
        ]
    )

    # Moving average windows
    mean = aggregates.select(pl.col("mean").rolling_mean(span)).collect()
    ci_low = interval.select(pl.col("low").rolling_mean(span)).collect()
    ci_high = interval.select(pl.col("high").rolling_mean(span)).collect()
    # margin = margin.rolling(span).mean()
    idx = aggregates.select("time/episodes").collect()
    (line,) = ax.plot(idx, mean.select("mean"))
    ax.fill_between(
        idx.to_series().to_numpy(),
        ci_low.select("low").to_series().to_numpy(),
        ci_high.select("high").to_series().to_numpy(),
        alpha=0.5,
        linewidth=0,
    )
    line.set_label(name)


def set_size(
    textwidth: float, fraction: float = 1, subplots: Tuple[int, int] = (1, 1)
) -> Tuple[float, float]:
    """Get the size (width, height) of the figure, given the textwidth.

    Setting the figure dimensions in matplotlib helps avoid scaling-related issues in
    LaTeX.

    See
    ---

    * https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    textwidth :
        Document textwidth or columnwidth in pts
    fraction :
            Fraction of the width which you wish the figure to occupy
    subplots :
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if textwidth == "thesis":
        width_pt = 426.79135
    elif textwidth == "beamer":
        width_pt = 307.28987
    else:
        width_pt = textwidth

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
