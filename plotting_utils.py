import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def prep_performance_by_model_dict(perf_excel_path):
    """ Converts the excel file in the specified path to a dictionary of sub dataframes of each model.
    
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
        results_dict.update({
            model: full_df[full_df["model"]==model]
        })
    return results_dict

def make_plot_two_algs(alg1, alg2, metric, sub_df_dict, ax, showxticklabels=True, showyticklabels=True, max_regret_val=None):
    x_df = sub_df_dict[alg1]
    y_df = sub_df_dict[alg2]
    test_comps = x_df["test_compound"].unique()
    x = [x_df[x_df["test_compound"]==i][metric].mean() for i in test_comps]
    y = [y_df[y_df["test_compound"]==i][metric].mean() for i in test_comps]
    color = {
        "regret":"#5ec962",
        "reciprocal_rank":"#21918c",
        "mean_reciprocal_rank":"#3b528b",
        "kendall_tau":"#440154"
    }
    ax.scatter(x, y, c=color[metric])
    if metric == "regret":
        if max_regret_val > 20 :
            jump = 10
        else : 
            jump = 5
        max_val = int(jump*(max_regret_val//jump+1)+1)
        ax.set_xlim(-max_val//20, max_val*1.05)
        ax.set_ylim(-max_val//20, max_val*1.05)
        ax.set_xticks(np.arange(0, max_val, jump))
        ax.set_yticks(np.arange(0, max_val, jump))
        if showxticklabels :
            ax.set_xticklabels(np.arange(0, max_val, jump))
        else : 
            ax.set_xticklabels([])
        if showyticklabels :
            ax.set_yticklabels(np.arange(0, max_val, jump))
        else : 
            ax.set_yticklabels([])
        ax.plot(np.arange(0,100), np.arange(0,100), ls='--', c="grey")
        p = np.arange(0,max_val*1.05)
        up = p + 5
        down = p - 5
        ax.fill_between(p, up, down, facecolor='grey', alpha=0.2)
    elif metric in ["mean_reciprocal_rank", "reciprocal_rank"]:
        ax.set_xlim(0,1.04)
        ax.set_ylim(0,1.04)
        ax.set_xticks([round(0.2*x,1) for x in range(6)])
        ax.set_yticks([round(0.2*x,1) for x in range(6)])
        if showxticklabels :
            ax.set_xticklabels([round(0.2*x,1) for x in range(6)])
        else :
            ax.set_xticklabels([""]*6)
        if showyticklabels:
            ax.set_yticklabels([round(0.2*x,1) for x in range(6)])
        else : 
            ax.set_yticklabels([""]*6)
        ax.plot(np.arange(0,1,0.01), np.arange(0,1,0.01), ls='--', c="grey")
    elif metric == "kendall_tau" :
        ax.set_xlim(-0.75,1)
        ax.set_ylim(-0.75,1)
        ax.set_xticks([round(0.25*x,2) for x in range(-3,5)])
        ax.set_yticks([round(0.25*x,2) for x in range(-3,5)])
        if showxticklabels :
            ax.set_xticklabels([round(0.25*x,2) for x in range(-3,5)])
        else :
            ax.set_xticklabels([""]*8)
        if showyticklabels:
            ax.set_yticklabels([round(0.25*x,2) for x in range(-3,5)])
        else : 
            ax.set_yticklabels([""]*8)
        ax.plot(np.arange(-0.75,1,0.01), np.arange(-0.75,1,0.01), ls='--', c="grey")
    ax.set_aspect("equal")
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)


def trellis_of_algs(
        sub_df_dict,
        list_of_algs,
        metric1,
        metric2,
        filename=None
    ):
    """ Prepares a trellis of all pairwise comparisons under two metrics.
    
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
        figsize=(3*len(list_of_algs),3*len(list_of_algs)),
        gridspec_kw = {"wspace":0.2, "hspace":0.2},
    )
    if metric1 == "regret" or metric2=="regret" :
        max_regret = 0
        for k, v in sub_df_dict.items():
            test_comps = v["test_compound"].unique()
            if k in list_of_algs :
                max_val = max([v[v["test_compound"]==i]["regret"].mean() for i in test_comps])
                if max_val > max_regret :
                    max_regret = max_val 
    else :
        max_regret=None
    for i, alg1 in enumerate(list_of_algs) :
        for j, alg2 in enumerate(list_of_algs) :
            showxticklabels = True
            showyticklabels = True
            if i > j :
                if i != len(list_of_algs) - 1:
                    showxticklabels = False
                if j > 0 :
                    showyticklabels = False
                make_plot_two_algs(alg2, alg1, metric1, sub_df_dict, ax[i,j], showxticklabels, showyticklabels, max_regret)
            elif i < j :
                if j >= i+2 :
                    showxticklabels = False
                    showyticklabels = False
                make_plot_two_algs(alg2, alg1, metric2, sub_df_dict, ax[i,j], showxticklabels, showyticklabels, max_regret)
            elif i==j :
                if i == 0 :
                    ax[0,0].set_ylabel(list_of_algs[0], fontsize=12, fontfamily="arial")
                    # remove ticks and labels for left axis
                    ax[j,i].tick_params(left=False, labelleft=False)
                    # make x axis invisible
                    ax[j,i].xaxis.set_visible(False)
                elif i == len(list_of_algs) - 1 :
                    ax[i, i].set_xlabel(
                        list_of_algs[-1], fontsize=12, fontfamily="arial"
                    )
                    # make y axis invisible
                    ax[j,i].yaxis.set_visible(False)
                    # remove ticks and labels for bottom axis
                    ax[j,i].tick_params(bottom=False, labelbottom=False)
                else :
                    # make x axis invisible
                    ax[j,i].xaxis.set_visible(False)
                    # make y axis invisible
                    ax[j,i].yaxis.set_visible(False)
                # makes the box invisible
                plt.setp(ax[j,i].spines.values(), visible=False)
            if j == 0 :
                ax[i,j].set_ylabel(alg1, fontsize=12, fontfamily="arial")
            if i == len(list_of_algs)-1 :
                ax[i,j].set_xlabel(alg2, fontsize=12, fontfamily="arial")
    formal_titles = {
        "regret":"Regret",
        "mean_reciprocal_rank":"Mean Reciprocal Rank",
        "reciprocal_rank":"Reciprocal Rank",
        "kendall_tau":"Kendall Tau",
    }
    fig.suptitle(formal_titles[metric2], fontsize=14, y=0.92, fontweight="bold", fontfamily="arial")
    fig.supylabel(formal_titles[metric1], fontsize=14, x=0.05, fontweight="bold", fontfamily="arial")
    plt.show()
