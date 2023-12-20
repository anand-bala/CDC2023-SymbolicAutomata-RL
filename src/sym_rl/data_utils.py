import re
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.axes import Axes


def list_data_files(logdir: Path, map_name: str, spec: str, method: str) -> List[Path]:
    """
    Given a map name, the specification, and the rewarding method, find the list of
    files where the evaluation data is stored. This list consists of all evaluations
    made across training iterations and random seeds. The caller should infer the
    training iteration from the file name.

    The output will be sorted by file name, i.e., by the iteration number.
    """
    eval_dir_re = re.compile(fr"{method}-seed\d{{5}}-eval")
    csv_file_re = re.compile(r"\d{7}.csv")

    logdir = Path(logdir)
    datadir = logdir / map_name / spec

    files = deque()
    for path in datadir.iterdir():
        if path.is_dir() and eval_dir_re.fullmatch(path.name):
            for file in path.iterdir():
                if file.is_file() and csv_file_re.fullmatch(file.name):
                    files.append(file)

    return list(sorted(files, key=lambda p: p.name))


def extract_data(data_files: List[Path]) -> pd.DataFrame:
    """From the list of CSV eval files, extract a dataframe"""

    data = pd.DataFrame()

    for f in data_files:
        iteration = int(f.stem)
        df: pd.DataFrame = pd.read_csv(
            f, index_col=False, usecols=["accepting"]
        )  # type: ignore
        df["training_iter"] = iteration
        data = pd.concat([data, df], ignore_index=True)

    if len(data) > 0:
        data.set_index("training_iter", inplace=True)
    return data


def load_data(logdir: Path, map_name: str, spec: str, method: str) -> pd.DataFrame:
    return extract_data(list_data_files(logdir, map_name, spec, method))


def create_data_table(col_name: str, *dfs: Tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """Given a column name, and a list of `(names, DataFrames)`, create a new DataFrame
    with `training_iter` as the index, and data from the columne in each input
    dataframe. The column names are set to the dataframe names.
    """

    data = pd.DataFrame()
    for name, df in dfs:
        if len(df) == 0:
            # Skip for unsupported methods.
            # E.g. TauMDP with the unbounded specifications.
            continue
        data[name] = df[col_name]
    return data


def _get_wilson_center(
    data: pd.Series, confidence=0.95
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Given a data of Bernoulli trials, get the Wilson center, the new count, and stddev for the data"""
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - (alpha / 2))

    new_x = data.groupby("training_iter").sum() + (z ** 2) / 2
    new_n = data.groupby("training_iter").count() + (z ** 2) / 2

    p = new_x / new_n
    q = 1 - p
    std = np.sqrt(p * q)
    return p, new_n, std


def plot_curve(
    name: str, ax: Axes, data: pd.Series, *, span=5, confidence=0.95, bernoulli=False
):
    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - (alpha / 2))

    aggregates = data.groupby("training_iter").agg(["mean", "std", "count"])

    mean, std, count = aggregates["mean"], aggregates["std"], aggregates["count"]
    if bernoulli:
        # Use Agresti–Coull interval
        # Essentially we need to adjust the mean and std deviations first
        p, new_count, new_std = _get_wilson_center(data, confidence=confidence)
        margin = z_critical * (new_std / np.sqrt(new_count))
        interval = (p - margin, p + margin)
    else:
        margin = z_critical * (std / np.sqrt(count))
        interval = (mean - margin, mean + margin)

    # Moving average windows
    mean = mean.rolling(span).mean()
    ci_low = interval[0].rolling(span).mean()
    ci_high = interval[1].rolling(span).mean()
    # margin = margin.rolling(span).mean()
    (line,) = ax.plot(mean.index, mean)
    ax.fill_between(
        mean.index,
        ci_low,
        ci_high,
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
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
