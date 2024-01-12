import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import rankdata


def prep_performance_by_model_dict(perf_excel_path):
    """Converts the excel file in the specified path to a dictionary of sub dataframes of each model.

    Parameters
    ----------
    perf_excel_path : str
        Path to the performance excel file.

    Returns
    -------
    results_dict : dict
        • key : model
        • val : pd.dataframe
    """
    full_df = pd.read_excel(perf_excel_path)
    results_dict = {}
    for model in full_df["model"].unique():
        results_dict.update({model: full_df[full_df["model"] == model]})
    return results_dict


def make_plot_two_algs(
    alg1,
    alg2,
    metric,
    sub_df_dict,
    ax,
    showxticklabels=True,
    showyticklabels=True,
    max_regret_val=None,
):
    x_df = sub_df_dict[alg1]
    y_df = sub_df_dict[alg2]
    test_comps = x_df["test_compound"].unique()
    x = [x_df[x_df["test_compound"] == i][metric].mean() for i in test_comps]
    y = [y_df[y_df["test_compound"] == i][metric].mean() for i in test_comps]
    color = {
        "regret": "#5ec962",
        "reciprocal_rank": "#21918c",
        "mean_reciprocal_rank": "#3b528b",
        "kendall_tau": "#440154",
    }
    ax.scatter(x, y, c=color[metric])
    if metric == "regret":
        if max_regret_val > 20:
            jump = 10
        else:
            jump = 5
        max_val = int(jump * (max_regret_val // jump + 1) + 1)
        ax.set_xlim(-max_val // 20, max_val * 1.05)
        ax.set_ylim(-max_val // 20, max_val * 1.05)
        ax.set_xticks(np.arange(0, max_val, jump))
        ax.set_yticks(np.arange(0, max_val, jump))
        if showxticklabels:
            ax.set_xticklabels(np.arange(0, max_val, jump))
        else:
            ax.set_xticklabels([])
        if showyticklabels:
            ax.set_yticklabels(np.arange(0, max_val, jump))
        else:
            ax.set_yticklabels([])
        ax.plot(np.arange(0, 100), np.arange(0, 100), ls="--", c="grey")
        p = np.arange(0, max_val * 1.05)
        up = p + 5
        down = p - 5
        ax.fill_between(p, up, down, facecolor="grey", alpha=0.2)
    elif metric in ["mean_reciprocal_rank", "reciprocal_rank"]:
        ax.set_xlim(0, 1.04)
        ax.set_ylim(0, 1.04)
        ax.set_xticks([round(0.2 * x, 1) for x in range(6)])
        ax.set_yticks([round(0.2 * x, 1) for x in range(6)])
        if showxticklabels:
            ax.set_xticklabels([round(0.2 * x, 1) for x in range(6)])
        else:
            ax.set_xticklabels([""] * 6)
        if showyticklabels:
            ax.set_yticklabels([round(0.2 * x, 1) for x in range(6)])
        else:
            ax.set_yticklabels([""] * 6)
        ax.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), ls="--", c="grey")
    elif metric == "kendall_tau":
        ax.set_xlim(-0.75, 1)
        ax.set_ylim(-0.75, 1)
        ax.set_xticks([round(0.25 * x, 2) for x in range(-3, 5)])
        ax.set_yticks([round(0.25 * x, 2) for x in range(-3, 5)])
        if showxticklabels:
            ax.set_xticklabels([round(0.25 * x, 2) for x in range(-3, 5)])
        else:
            ax.set_xticklabels([""] * 8)
        if showyticklabels:
            ax.set_yticklabels([round(0.25 * x, 2) for x in range(-3, 5)])
        else:
            ax.set_yticklabels([""] * 8)
        ax.plot(np.arange(-0.75, 1, 0.01), np.arange(-0.75, 1, 0.01), ls="--", c="grey")
    ax.set_aspect("equal")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)


def trellis_of_algs(sub_df_dict, list_of_algs, metric1, metric2, filename=None):
    """Prepares a trellis of all pairwise comparisons under two metrics.

    Parameters
    ----------
    sub_df_dict : dict
        Output of the function above.
    list_of_algs : list
        List of algorithms to compare.
    metric1, metric2 : str {'regret', 'reciprocal_rank', 'mean_reciprocal_rank', 'kendall_tau'}
        Metrics to compare. Metrics 1 and 2 are drawn out at the bottom left half, top right half, respectively.
    filename: None or str
        Path to save the resulting plot.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(
        len(list_of_algs),
        len(list_of_algs),
        figsize=(3 * len(list_of_algs), 3 * len(list_of_algs)),
        gridspec_kw={"wspace": 0.2, "hspace": 0.2},
    )
    if metric1 == "regret" or metric2 == "regret":
        max_regret = 0
        for k, v in sub_df_dict.items():
            test_comps = v["test_compound"].unique()
            if k in list_of_algs:
                max_val = max(
                    [v[v["test_compound"] == i]["regret"].mean() for i in test_comps]
                )
                if max_val > max_regret:
                    max_regret = max_val
    else:
        max_regret = None
    for i, alg1 in enumerate(list_of_algs):
        for j, alg2 in enumerate(list_of_algs):
            showxticklabels = True
            showyticklabels = True
            if i > j:
                if i != len(list_of_algs) - 1:
                    showxticklabels = False
                if j > 0:
                    showyticklabels = False
                make_plot_two_algs(
                    alg2,
                    alg1,
                    metric1,
                    sub_df_dict,
                    ax[i, j],
                    showxticklabels,
                    showyticklabels,
                    max_regret,
                )
            elif i < j:
                if j >= i + 2:
                    showxticklabels = False
                    showyticklabels = False
                make_plot_two_algs(
                    alg2,
                    alg1,
                    metric2,
                    sub_df_dict,
                    ax[i, j],
                    showxticklabels,
                    showyticklabels,
                    max_regret,
                )
            elif i == j:
                if i == 0:
                    ax[0, 0].set_ylabel(
                        list_of_algs[0], fontsize=12, fontfamily="arial"
                    )
                    # remove ticks and labels for left axis
                    ax[j, i].tick_params(left=False, labelleft=False)
                    # make x axis invisible
                    ax[j, i].xaxis.set_visible(False)
                elif i == len(list_of_algs) - 1:
                    ax[i, i].set_xlabel(
                        list_of_algs[-1], fontsize=12, fontfamily="arial"
                    )
                    # make y axis invisible
                    ax[j, i].yaxis.set_visible(False)
                    # remove ticks and labels for bottom axis
                    ax[j, i].tick_params(bottom=False, labelbottom=False)
                else:
                    # make x axis invisible
                    ax[j, i].xaxis.set_visible(False)
                    # make y axis invisible
                    ax[j, i].yaxis.set_visible(False)
                # makes the box invisible
                plt.setp(ax[j, i].spines.values(), visible=False)
            if j == 0:
                ax[i, j].set_ylabel(alg1, fontsize=12, fontfamily="arial")
            if i == len(list_of_algs) - 1:
                ax[i, j].set_xlabel(alg2, fontsize=12, fontfamily="arial")
    formal_titles = {
        "regret": "Regret",
        "mean_reciprocal_rank": "Mean Reciprocal Rank",
        "reciprocal_rank": "Reciprocal Rank",
        "kendall_tau": "Kendall Tau",
    }
    fig.suptitle(
        formal_titles[metric2],
        fontsize=14,
        y=0.92,
        fontweight="bold",
        fontfamily="arial",
    )
    fig.supylabel(
        formal_titles[metric1],
        fontsize=14,
        x=0.05,
        fontweight="bold",
        fontfamily="arial",
    )
    plt.show()


### For Active learning analysis
def AL_trellis(df_to_plot, rpc_df, rfr_df, ymin, ymax):
    """Draws a trellis of AL performances of the first 25 trials."""
    fig, ax = plt.subplots(
        nrows=5, ncols=5, sharex=True, sharey=True, figsize=(15, 15)
    )  # 12,12
    for i in range(25):
        row = i // 5
        col = i % 5
        sns.lineplot(
            df_to_plot[df_to_plot["Evaluation Iteration"] == i],
            x="Substrates Sampled",
            y="Reciprocal Rank",
            hue="Strategy",
            style="Strategy",
            markers=True,
            ax=ax[row, col],
            palette="viridis",
            alpha=0.7,
        )
        ax[row, col].plot(
            np.arange(6, 38, 2),
            [rpc_df[rpc_df["Evaluation Iteration"] == i]["Reciprocal Rank"]] * 16,
            color="orange",
            alpha=0.5,
            ls="--",
        )
        if rfr_df is not None:
            ax[row, col].plot(
                np.arange(6, 38, 2),
                [rfr_df[rfr_df["Evaluation Iteration"] == i]["Reciprocal Rank"]] * 16,
                color="grey",
                alpha=0.5,
                ls="--",
            )
        if row == 0 and col == 4:
            ax[row, col].legend(bbox_to_anchor=(1.01, 0.99))
        else:
            ax[row, col].get_legend().remove()
        ax[row, col].set_ylim(ymin, ymax)
        ax[row, col].set_yticks(
            [round(0.1 * x, 1) for x in range(int(10 * ymin), int(10 * ymax) + 1)]
        )
        if row == 4 and col == 2:
            ax[row, col].set_xlabel("Number of Sampled Substrates", fontsize=12)
        else:
            ax[row, col].set_xlabel("")
        if row == 2 and col == 0:
            ax[row, col].set_ylabel("Mean Reciprocal Rank", fontsize=12)
        else:
            ax[row, col].set_ylabel("")
        if col == 0:
            ax[row, col].set_yticks(
                [round(0.1 * x, 1) for x in range(int(10 * ymin), int(10 * ymax) + 1)]
            )

        for axis in ["top", "bottom", "left", "right"]:
            ax[row, col].spines[axis].set_linewidth(2)
    plt.show()


def AL_average(df_to_plot, rpc_df, rfr_df, xmin, xmax):
    fig, ax = plt.subplots()
    sns.lineplot(
        df_to_plot,
        x="Substrates Sampled",
        y="Reciprocal Rank",
        hue="Strategy",
        style="Strategy",
        markers=True,
        palette="viridis",
    )
    ax.plot(
        np.arange(xmin, xmax + 1),
        [rpc_df["Reciprocal Rank"].mean()] * (xmax + 1 - xmin),
        color="orange",
        alpha=0.7,
        ls="--",
    )
    if rfr_df is not None:
        ax.plot(
            np.arange(xmin, xmax + 1),
            [rfr_df["Reciprocal Rank"].mean()] * (xmax + 1 - xmin),
            color="grey",
            alpha=0.7,
            ls="--",
        )
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.set_xlim(xmin, xmax)
    ax.legend(bbox_to_anchor=(1.01, 0.99))
    plt.show()


### Borrowed from scikit_posthocs._plotting.py due to unknown error
from typing import Union, List, Tuple, Dict, Set

import numpy as np
from matplotlib.axes import SubplotBase
from matplotlib import pyplot
from pandas import DataFrame, Series


def sign_array(p_values: Union[List, np.ndarray], alpha: float = 0.05) -> np.ndarray:
    """Significance array.

    Converts an array with p values to a significance array where
    0 is False (not significant), 1 is True (significant),
    and -1 is for diagonal elements.

    Parameters
    ----------
    p_values : Union[List, np.ndarray]
        Any object exposing the array interface and containing
        p values.

    alpha : float = 0.05
        Significance level. Default is 0.05.

    Returns
    -------
    result : numpy.ndarray
        Array where 0 is False (not significant), 1 is True (significant),
        and -1 is for diagonal elements.

    Examples
    --------
    >>> p_values = np.array([[ 1.        ,  0.00119517,  0.00278329],
                             [ 0.00119517,  1.        ,  0.18672227],
                             [ 0.00278329,  0.18672227,  1.        ]])
    >>> ph.sign_array(p_values)
    array([[1, 1, 1],
           [1, 1, 0],
           [1, 0, 1]])
    """
    p_values = np.array(p_values)
    p_values[p_values > alpha] = 0
    p_values[(p_values < alpha) & (p_values > 0)] = 1
    np.fill_diagonal(p_values, 1)

    return p_values


def _find_maximal_cliques(adj_matrix: DataFrame) -> List[Set]:
    """Wrapper function over the recursive Bron-Kerbosch algorithm.

    Will be used to find points that are under the same crossbar in critical
    difference diagrams.

    Parameters
    ----------
    adj_matrix : pandas.DataFrame
        Binary matrix with 1 if row item and column item do NOT significantly
        differ. Values in the main diagonal are not considered.

    Returns
    -------
    list[set]
        Largest fully connected subgraphs, represented as sets of indices of
        adj_matrix.

    Raises
    ------
    ValueError
        If the input matrix is empty or not symmetric.
        If the input matrix is not binary.

    """
    if (adj_matrix.index != adj_matrix.columns).any():
        raise ValueError("adj_matrix must be symmetric, indices do not match")
    if not adj_matrix.isin((0, 1)).values.all():
        raise ValueError("Input matrix must be binary")
    if adj_matrix.empty or not (adj_matrix.T == adj_matrix).values.all():
        raise ValueError("Input matrix must be non-empty and symmetric")

    result = []
    _bron_kerbosch(
        current_clique=set(),
        candidates=set(adj_matrix.index),
        visited=set(),
        adj_matrix=adj_matrix,
        result=result,
    )
    return result


def _bron_kerbosch(
    current_clique: Set,
    candidates: Set,
    visited: Set,
    adj_matrix: DataFrame,
    result: List[Set],
) -> None:
    """Recursive algorithm to find the maximal fully connected subgraphs.

    See [1]_ for more information.

    Parameters
    ----------
    current_clique : set
        A set of vertices known to be fully connected.
    candidates : set
        Set of vertices that could potentially be added to the clique.
    visited : set
        Set of vertices already known to be part of another previously explored
        clique, that is not current_clique.
    adj_matrix : pandas.DataFrame
        Binary matrix with 1 if row item and column item do NOT significantly
        differ. Diagonal must be zeroed.
    result : list[set]
        List where to append the maximal cliques.

    Returns
    -------
    None

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    """
    while candidates:
        v = candidates.pop()
        _bron_kerbosch(
            current_clique | {v},
            # Restrict candidate vertices to the neighbors of v
            {n for n in candidates if adj_matrix.loc[v, n]},
            # Restrict visited vertices to the neighbors of v
            {n for n in visited if adj_matrix.loc[v, n]},
            adj_matrix,
            result,
        )
        visited.add(v)

    # We do not need to report a clique if a children call aready did it.
    if not visited:
        # If this is not a terminal call, i.e. if any clique was reported.
        result.append(current_clique)


def critical_difference_diagram(
    ranks: Union[dict, Series],
    sig_matrix: DataFrame,
    *,
    ax: SubplotBase = None,
    label_fmt_left: str = "{label} ({rank:.2g})",
    label_fmt_right: str = "({rank:.2g}) {label}",
    label_props: dict = None,
    marker_props: dict = None,
    elbow_props: dict = None,
    crossbar_props: dict = None,
    text_h_margin: float = 0.01
) -> Dict[str, list]:
    """Plot a Critical Difference diagram from ranks and post-hoc results.

    The diagram arranges the average ranks of multiple groups on the x axis
    in order to facilitate performance comparisons between them. The groups
    that could not be statistically deemed as different are linked by a
    horizontal crossbar [1]_, [2]_.

    ::

                      rank markers
         X axis ---------O----O-------------------O-O------------O---------
                         |----|                   | |            |
                         |    |                   |---crossbar---|
                clf1 ----|    |                   | |            |---- clf3
                clf2 ---------|                   | |----------------- clf4
                                                  |------------------- clf5
                    |____|
                text_h_margin

    In the drawing above, the two crossbars indicate that clf1 and clf2 cannot
    be statistically differentiated, the same occurring between clf3, clf4 and
    clf5. However, clf1 and clf2 are each significantly lower ranked than clf3,
    clf4 and clf5.

    Parameters
    ----------
    ranks : dict or Series
        Indicates the rank value for each sample or estimator (as keys or index).

    sig_matrix : DataFrame
        The corresponding p-value matrix outputted by post-hoc tests, with
        indices matching the labels in the ranks argument.

    ax : matplotlib.SubplotBase, optional
        The object in which the plot will be built. Gets the current Axes
        by default (if None is passed).

    label_fmt_left : str, optional
        The format string to apply to the labels on the left side. The keywords
        label and rank can be used to specify the sample/estimator name and
        rank value, respectively, by default '{label} ({rank:.2g})'.

    label_fmt_right : str, optional
        The same, but for the labels on the right side of the plot.
        By default '({rank:.2g}) {label}'.

    label_props : dict, optional
        Parameters to be passed to pyplot.text() when creating the labels,
        by default None.

    marker_props : dict, optional
        Parameters to be passed to pyplot.scatter() when plotting the rank
        markers on the axis, by default None.

    elbow_props : dict, optional
        Parameters to be passed to pyplot.plot() when creating the elbow lines,
        by default None.

    crossbar_props : dict, optional
        Parameters to be passed to pyplot.plot() when creating the crossbars
        that indicate lack of statistically significant difference. By default
        None.

    text_h_margin : float, optional
        Space between the text labels and the nearest vertical line of an
        elbow, by default 0.01.

    Returns
    -------
    dict[str, list[matplotlib.Artist]]
        Lists of Artists created.

    Examples
    --------
    See the :doc:`/tutorial`.

    References
    ----------
    .. [1] Demšar, J. (2006). Statistical comparisons of classifiers over multiple
            data sets. The Journal of Machine learning research, 7, 1-30.

    .. [2] https://mirkobunse.github.io/CriticalDifferenceDiagrams.jl/stable/
    """
    import cycler, matplotlib
    color = pyplot.cm.viridis(np.linspace(0, 1, 9))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    elbow_props = elbow_props or {}
    marker_props = {"zorder": 3, **(marker_props or {})}
    label_props = {"va": "center", **(label_props or {})}
    crossbar_props = {
        "color": "k",
        "zorder": 3,
        "linewidth": 2,
        **(crossbar_props or {}),
    }

    ax = ax or pyplot.gca()
    ax.yaxis.set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", labelsize=5)
    ax.xaxis.set_ticks_position("top")
    ax.spines["top"].set_position("zero")

    # lists of artists to be returned
    markers = []
    elbows = []
    labels = []
    crossbars = []

    # True if pairwise comparison is NOT significant
    adj_matrix = DataFrame(
        1 - sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )

    ranks = Series(ranks)  # Standardize if ranks is dict
    points_left, points_right = np.array_split(ranks.sort_values(), 2)

    # Sets of points under the same crossbar
    crossbar_sets = _find_maximal_cliques(adj_matrix)

    # Sort by lowest rank and filter single-valued sets
    crossbar_sets = sorted(
        (x for x in crossbar_sets if len(x) > 1), key=lambda x: ranks[list(x)].min()
    )

    # Create stacking of crossbars: for each level, try to fit the crossbar,
    # so that it does not intersect with any other in the level. If it does not
    # fit in any level, create a new level for it.
    crossbar_levels: list[list[set]] = []
    for bar in crossbar_sets:
        for level, bars_in_level in enumerate(crossbar_levels):
            if not any(bool(bar & bar_in_lvl) for bar_in_lvl in bars_in_level):
                ypos = -level - 1
                bars_in_level.append(bar)
                break
        else:
            ypos = -len(crossbar_levels) - 1
            crossbar_levels.append([bar])

        crossbars.append(
            ax.plot(
                # Adding a separate line between each pair enables showing a
                # marker over each elbow with crossbar_props={'marker': 'o'}.
                [ranks[i] for i in bar],
                [ypos] * len(bar),
                **crossbar_props,
            )
        )

    lowest_crossbar_ypos = -len(crossbar_levels)

    def plot_items(points, xpos, label_fmt, label_props):
        """Plot each marker + elbow + label."""
        ypos = lowest_crossbar_ypos - 1
        for label, rank in points.items():
            elbow, *_ = ax.plot(
                [xpos, rank, rank],
                [ypos, ypos, 0],
                **elbow_props,
            )
            elbows.append(elbow)
            curr_color = elbow.get_color()
            markers.append(ax.scatter(rank, 0, **{"color": curr_color, **marker_props}))
            labels.append(
                ax.text(
                    xpos,
                    ypos,
                    label_fmt.format(label=label, rank=rank),
                    **{"color": curr_color, **label_props},
                )
            )
            ypos -= 1

    plot_items(
        points_left,
        xpos=points_left.iloc[0] - text_h_margin,
        label_fmt=label_fmt_left,
        label_props={"ha": "right", **label_props},
    )
    plot_items(
        points_right[::-1],
        xpos=points_right.iloc[-1] + text_h_margin,
        label_fmt=label_fmt_right,
        label_props={"ha": "left", **label_props},
    )

    return {
        "markers": markers,
        "elbows": elbows,
        "labels": labels,
        "crossbars": crossbars,
    }


def plot_rr_heatmap(rr_array, model_list, dataset_list, hline_pos=[5,9], vline_pos=[1,2,5], save=False):
    """ Plots a heatmap of reciprocal ranks (RR) achieved by each algorithm in each dataset.
    The annotated numbers are the RR values, while the colors denote the rank of each algorithm 
    in each dataset.
    
    Parameters
    ----------
    rr_array : np.ndarray of shape (n_datasets, n_models)
        Reciprocal rank values to plot.
    model_list : list of str 
        Names of algorithms.
    dataset_list : list of str
        Names of datasets.
    save : False or str
        Whether to save the resulting plot. If str, will be used as filename.
    
    Returns
    -------
    None
    """
    # Version 2 - color by rank in each dataset, annotate the RR values
    # fig, ax = plt.subplots()
    ordered_rr_rank_by_dataset = rankdata(rr_array, axis=1)
    n_models = len(model_list)
    cmap = sns.color_palette("viridis", n_models)
    vmap = {i: 1+n_models-i for i in range(1,1+n_models)}
    ax = sns.heatmap(ordered_rr_rank_by_dataset, cmap=cmap, annot=rr_array)
    # get colorbar from seaborn heatmap
    colorbar = ax.collections[0].colorbar
    # define discrete intervals for colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n_models) + r * i / (n_models) for i in range(n_models)])
    colorbar.set_ticklabels(list(vmap.values()))
    colorbar.set_label("Model's rank", fontdict={"fontfamily":"arial", "fontsize":10})

    for v in vline_pos :
        ax.axvline(v,0,1, c="white", lw=0.5)
    for h in hline_pos : 
        ax.axhline(h,0,1, c="white", lw=0.5)
    
    ax.set_xticklabels(model_list, fontdict={"fontfamily":"arial", "fontsize":10})
    ax.set_yticklabels(dataset_list, fontdict={"fontfamily":"arial", "fontsize":10}, rotation=0)
    ax.set_xlabel("Models", fontdict={"fontfamily":"arial", "fontsize":12})
    ax.set_ylabel("Datasets", fontdict={"fontfamily":"arial", "fontsize":12})
    if type(save) == str:
        plt.savefig(f"figures/{save}", dpi=300, format="svg")
