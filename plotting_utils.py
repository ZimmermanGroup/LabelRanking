import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from copy import deepcopy
import seaborn as sns
from scipy.stats.mstats import rankdata
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from itertools import combinations

class Analyzer():
    """ Analyzing results of all datasets in a consistent manner.
    We assume that all 11 datasets are being analyzed simultaneously.

    Parameters
    ----------
    feature : str
        Featurization used for the models
    models : list of str
        Which models are used.
    """
    def __init__(self, feature, models, n_rem_rxns=0) :
        self.feature = feature
        self.models = models
        self.n_rem_rxns = n_rem_rxns
        if self.n_rem_rxns == 0 :
            self._filename_appendix = ".xlsx"
        else :
            self._filename_appendix = f"_rem{self.n_rem_rxns}rxns.xlsx"
        # Loading the datasets
        self.deoxy_perf_df = pd.read_excel(f"performance_excels/deoxy/{self.feature}_base_None"+self._filename_appendix)
        self.amine_perf_df = pd.read_excel(f"performance_excels/natureHTE/{self.feature}_amine_None"+self._filename_appendix)
        self.amide_perf_df = pd.read_excel(f"performance_excels/natureHTE/{self.feature}_amide_None"+self._filename_appendix)
        self.sulfon_perf_df = pd.read_excel(f"performance_excels/natureHTE/{self.feature}_sulfonamide_None"+self._filename_appendix)
        self.thiol_perf_df = pd.read_excel(f"performance_excels/natureHTE/{self.feature}_thiol_None"+self._filename_appendix)
        self.whole_amine_perf_df = pd.read_excel(f"performance_excels/scienceMALDI/{self.feature}_whole_amine_None"+self._filename_appendix)
        self.whole_bromide_perf_df = pd.read_excel(f"performance_excels/scienceMALDI/{self.feature}_whole_bromide_None"+self._filename_appendix)

    def _get_different_sf_index_for_deoxy(self, deoxy_df):
        """ Result files of deoxyfluorination datasets consist of the 5 different sulfonyl fluorides altogether.
        FInds the indices where results of each sulfonyl fluoride starts.
        
        Parameters
        ----------
        deoxy_df : pd.DataFrame
            result dataframe of deoxy dataset.
        
        Returns
        -------
        new_comp_starts_at : list of int
            The indices.
        """
        new_comp_starts_at = [0]
        rfr_inds = deoxy_df[deoxy_df["model"]=="RFR"].index.tolist()
        for i, ind in enumerate(rfr_inds) :
            if i> 0 :
                if ind - rfr_inds[i-1] > 1 :
                    new_comp_starts_at.append(ind)
        return new_comp_starts_at
    
    def _update_perf_dict(self, dict_to_update, raw_df, model_name, dataset_name) :
        dict_to_update["model"].append(model_name)
        dict_to_update["dataset"].append(dataset_name)
        dict_to_update["average reciprocal rank"].append(
            raw_df[raw_df["model"]==model_name]["reciprocal_rank"].mean()
        )
        if "average mean reciprocal rank" in dict_to_update.keys():
            dict_to_update["average mean reciprocal rank"].append(
                raw_df[raw_df["model"]==model_name]["mean_reciprocal_rank"].mean()
            )
        dict_to_update["average kendall tau"].append(
            raw_df[raw_df["model"]==model_name]["kendall_tau"].mean()
        )
        dict_to_update["regret"].append(
            raw_df[raw_df["model"]==model_name]["regret"].mean()
        )

    @property
    def avg_perf_df(self):
        """ Collects all results into one dataframe."""
        
        avg_perf_dict = {
            "model":[],
            "dataset":[],
            "average reciprocal rank":[],
            "average kendall tau":[],
            "regret":[]
        }
        deoxy_start_inds = self._get_different_sf_index_for_deoxy(self.deoxy_perf_df)
        for i, start_ind in enumerate(deoxy_start_inds) :
            if i!= 4:
                deoxy_sub_df = self.deoxy_perf_df.iloc[start_ind:deoxy_start_inds[i+1]]
            else :
                deoxy_sub_df = self.deoxy_perf_df.iloc[start_ind:]
            for model in self.models :
                self._update_perf_dict(avg_perf_dict, deoxy_sub_df, model, f"Deoxy-sulfonyl fluoride {i}")

        nature_HTE_perf_dfs = [self.amine_perf_df, self.amide_perf_df, self.sulfon_perf_df, self.thiol_perf_df]
        nature_HTE_names = ["amine", "amide", "sulfonamide", "thiol"]
        for model in self.models :
            for dataset_name, perf_df in zip(nature_HTE_names, nature_HTE_perf_dfs) :
                self._update_perf_dict(avg_perf_dict, perf_df, model, f"Nature-{dataset_name}")
                
        scienceMALDI_perf_dfs = [self.whole_amine_perf_df, self.whole_bromide_perf_df]
        scienceMALDI_names = ["whole amine", "whole bromide"]
        for model in self.models :
            for dataset_name, perf_df in zip(scienceMALDI_names, scienceMALDI_perf_dfs) :
                self._update_perf_dict(avg_perf_dict, perf_df, model, f"Science-{dataset_name}")

        self._avg_perf_df = pd.DataFrame(avg_perf_dict)
        return self._avg_perf_df
    

class MoreConditionAnalyzer(Analyzer):
    """ Used to analyze the results from datasets with more than 10 reaction conditions to choose from."""
    def __init__(self, feature, datasets, models, n_rem_rxns=0) :
        """ 
        Parameters
        ----------
        n_rem_rxns : int or list
            if list, we assume that the order is in informer, ullmann and borylation.
        """
        if type(n_rem_rxns) == int :
            super().__init__(feature, models, n_rem_rxns)
        else : 
            super().__init__(feature, models, 0)
        if self.feature == "fp" : n = 3
        elif self.feature == "desc" : n = 2
        # if type(n_rem_rxns) == int:
        #     appendix = [self._filename_appendix] * n
        # elif type(n_rem_rxns) == list :
        #     appendix = [f"_rem{x}rxns.xlsx" for x in n_rem_rxns]
        self.datasets = datasets
            
        # Loading the datasets
        df_list = []
        for i, dataset in enumerate(self.datasets) :
            if dataset == "informer" :
                df_list.append(pd.read_excel(f"performance_excels/informer/{self.feature}_catalyst_ratio_None.xlsx")) #+appendix[0]))
                self.informer_ind = i
            elif dataset == "ullmann" :
                df_list.append(pd.read_excel(f"performance_excels/ullmann/{self.feature}_None_None.xlsx")) #+appendix[1]))
            elif dataset == "borylation" :
                df_list.append(pd.read_excel(f"performance_excels/borylation/{self.feature}_None_None.xlsx")) #+appendix[2]))
            elif dataset == "aryl_borylation":
                meoh_df = pd.read_excel(f"performance_excels/aryl_borylation/{self.feature}_MeOH_None.xlsx")
                etoh_df = pd.read_excel(f"performance_excels/aryl_borylation/{self.feature}_EtOH_None.xlsx")
                df_list.append(meoh_df)
                df_list.append(etoh_df)
                self.datasets.insert(i+1, "aryl_borylation_etoh")
        self.df_list = df_list

    def _get_different_amine_ratio_index_for_informer(self, informer_df) :
        """Result files of informer datasets consist of 2 different amine_ratio values combined.
        Finds the indices where results of a new amine_ratio starts.
        
        Parameters
        ----------
        informer_df : pd.DataFrame
            result dataframe of informer dataset.
        
        Returns
        -------
        new_comp_starts_at : int
            Index where the results using the new sub-dataset starts at.
        """
        rfr_inds = informer_df[informer_df["model"] == "RFR"].index.tolist()
        for i, ind in enumerate(rfr_inds) :
            if i > 0 :
                if ind - rfr_inds[i-1] > 1 :
                    new_comp_starts_at = ind
                    break
        return new_comp_starts_at
    
    @property
    def avg_perf_df(self) :
        """ Collects all results into one dataframe."""
        avg_perf_dict = {
            "model":[],
            "dataset":[],
            "average reciprocal rank":[],
            "average mean reciprocal rank":[],
            "average kendall tau":[],
            "regret":[]
        }
        dfs = []
        names = []
        for name, df in zip(self.datasets, self.df_list) :
            if name == "informer" :
                divider = self._get_different_amine_ratio_index_for_informer(df)
                dfs.extend([df.iloc[:divider], df.iloc[divider:]])
                names.extend(["Informer 1", "Informer 2"])
            elif name == "ullmann" :
                dfs.append(df)
                names.append("Ullmann")
            elif name == "borylation" :
                dfs.append(df)
                names.append("Borylation")
            elif "aryl_borylation" in name :
                dfs.append(df)
                if "etoh" in name :
                    names.append("Aryl Borylation in EtOH")
                else :
                    names.append("Aryl Borylation in MeOH")
     
        for df, name in zip(dfs, names):
            for model in self.models :
                self._update_perf_dict(avg_perf_dict, df, model, name)
        self._avg_perf_df = pd.DataFrame(avg_perf_dict)
        return self._avg_perf_df


def get_rr_kt_tables(avg_perf_df, ordered_cols):
    """ Separated out from the analyzer because arrays from fingerprints need to be combined with descriptors in some cases."""
    ### Reformating the dataframe
    rr_table = pd.pivot_table(avg_perf_df, values="average reciprocal rank", index="dataset", columns="model")
    mrr_table = pd.pivot_table(avg_perf_df, values="average mean reciprocal rank", index="dataset", columns="model")
    kt_table = pd.pivot_table(avg_perf_df, values="average kendall tau", index="dataset", columns="model")
    return rr_table[ordered_cols], kt_table[ordered_cols], mrr_table[ordered_cols]

def run_friedman_tests(rr_Table, kt_table, models):
    """ Run Friedman rank tests on the ranks of each algorithm across all datasets, measured by 
    either reciprocal rank or kendall tau.
    
    Parameters
    ----------
    rr_Table : pd.DataFrame
        Table of average reciprocal rank values across datasets in each dataset.
    kt_table : pd.DataFrame
        Table of average kendall tau values across datasets in each dataset.
    models : list of str
        Models to compare between.

    Returns
    -------
    rr_pvalue, kt_pvalue : tuple of floats
        p values obtained for each metric.
    """
    ### Getting the rank of algorithms for each dataset in both metrics
    rr_rank_by_dataset = rr_Table.shape[1] + 1 - rankdata(rr_Table, axis=1)
    kt_rank_by_dataset = kt_table.shape[1] + 1 - rankdata(kt_table, axis=1)

    ### Friedman test
    rr_friedman_results = friedmanchisquare(
        *(rr_rank_by_dataset[:, x] for x in range(len(models)))
    )
    kt_friedman_results = friedmanchisquare(
        *(kt_rank_by_dataset[:, x] for x in range(len(models)))
    )
    return rr_friedman_results.pvalue, kt_friedman_results.pvalue

def plot_bonferroni_dunn(table, cols, save):
    rank_by_dataset = table.shape[1] + 1 - rankdata(table, axis=1)
    bonferroni_dunn_test_results = sp.posthoc_dunn(
        table.unstack().reset_index(name="average rank"), 
        val_col="average rank",
        group_col="model",
        p_adjust="bonferroni"
    )
    combined_rr_rank_dict = {x:r for x, r in zip(cols, np.average(rank_by_dataset, axis=0))}
    plt.set_cmap("viridis")
    plt.figure(figsize=(3.3,1), dpi=300)
    _ = critical_difference_diagram(
        combined_rr_rank_dict, 
        bonferroni_dunn_test_results, 
        label_props={"color":"k", "fontfamily":"arial", "fontsize":5}
    )
    if type(save) == str:
        plt.savefig(f"figures/{save}", dpi=300, format="svg")

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


def load_result(dataset, model1, model2):
    if "deoxy" in dataset :
        filename = "performance_excels/deoxy/desc_base_None.xlsx"
        kth = int(dataset[-1])
    elif "nature" in dataset :
        filename = f"performance_excels/natureHTE/desc_{dataset.split('_')[1]}_None.xlsx"
    elif "science" in dataset :
        filename = f"performance_excels/scienceMALDI/desc_whole_{dataset.split('_')[1]}_None.xlsx"
    elif "ullmann" in dataset :
        filename = f"performance_excels/ullmann/desc_None_None.xlsx"
    result_df = pd.read_excel(filename)
    model1_df = result_df[result_df["model"]==model1]
    model2_df = result_df[result_df["model"]==model2]
    # The deoxy dataset records results for all sulfonyl fluorides in a single file.
    if "deoxy" in dataset :
        assert model1_df.shape[0] == 160
        assert model2_df.shape[0] == 160
        model1_df = model1_df.iloc[32*(kth-1):32*kth]
        model2_df = model2_df.iloc[32*(kth-1):32*kth]
    return model1_df, model2_df

def compare_specific_recommendations(dataset, model1, model2, vmax=25, vmin=-20, yticks=[-20,-10,0,10,20], save=False):
    """Draws a heatmap of the rank of the suggestions made by each model.
    
    Parameters
    ----------
    dataset: str {deoxy1, deoxy2, deoxy3, deoxy4, deoxy5, nature_amide, nature_amine, nature_sulfonamide, nature_thiol, science_amine, science_bromide}
        Name of dataset to compare
    model1, model2: str
        Name of models to compare.
    save : bool
        Whether to save the plot.

    Returns
    -------
    None
    """
    # 1) Load the result excel
    model1_df, model2_df = load_result(dataset, model1, model2)
    # 2) Go through each tested substrate
    # Since RFR has been recorded multiple times, we will jist evaluate once
    counts = np.zeros((4,4)) # row : model1, col: model2
    bins = [0.33, 0.5, 1]
    for i in range(min(model1_df.shape[0], model2_df.shape[0])) :
        mrr_vals = np.array([model1_df.iloc[i,2], model2_df.iloc[i,2]])
        counts[np.digitize(mrr_vals, bins)[0], np.digitize(mrr_vals, bins)[1]] += 1

    # 3) Draw the heatmap
    fig, ax = plt.subplots(figsize=(2,3.6), nrows=2, tight_layout=True)
    sns.heatmap(counts, annot=True, square=True, cbar=False, annot_kws={"fontsize":6, "fontfamily":"arial"}, ax=ax[0], cmap="viridis")
    ax[0].set_xlabel(f"Rank of {model2} selection", fontdict={"fontfamily":"arial", "fontsize":8, "fontweight":"bold"})
    ax[0].set_ylabel(f"Rank of {model1} selection", fontdict={"fontfamily":"arial", "fontsize":8, "fontweight":"bold"})
    ax[0].set_xticklabels(5-np.arange(1,5), fontdict={"fontfamily":"arial", "fontsize":6})
    ax[0].set_yticklabels(5-np.arange(1,5), fontdict={"fontfamily":"arial", "fontsize":6})
    ax[0].xaxis.tick_top()
    ax[0].xaxis.set_label_position('top')

    # 4) Getting regrets
    regrets = []
    for i in range(min(model1_df.shape[0], model2_df.shape[0])) :
        regret_vals = np.array([model1_df.iloc[i,4], model2_df.iloc[i,4]])
        if regret_vals[0] != regret_vals[1] :
            regrets.append(regret_vals[0] - regret_vals[1])
    regrets = np.sort(np.array(regrets))[::-1]

    # 5) Plotting regrets
    ax[1].bar(np.arange(len(regrets)), regrets, width=0.9, color=["#5ec962"]*len(np.where(regrets>0)[0])+["#3b528b"]*len(np.where(regrets <0)[0]))
    # xticks, yticks = ax[1].get_xticks, ax[1].get_yticks()
    ax[1].set_ylim(vmin, vmax)
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(yticks, fontdict={"fontfamily":"arial", "fontsize":6})
    # ax[1].set_yticklabels([round(y) for y in yticks], fontdict={"fontfamily":"arial", "fontsize":6})
    ax[1].set_xticks(np.arange(len(regrets)))
    ax[1].set_xticklabels(np.arange(1, len(regrets)+1), fontdict={"fontfamily":"arial", "fontsize":6})
    ax[1].set_xlabel("Differently predicted substrates", fontdict={"fontfamily":"arial", "fontsize":8, "fontweight":"bold"})
    ax[1].set_ylabel(f"Yield({model2}) - Yield({model1}) (%)", fontdict={"fontfamily":"arial", "fontsize":8, "fontweight":"bold"})
    ax[1].axhline(0, 0, 1, color="k", linewidth=0.5)
    for axis in ["top", "bottom", "left", "right"]:
        ax[1].spines[axis].set_linewidth(1.5)

    if save :
        plt.savefig(f"figures/{dataset}_{model1}_{model2}_v2.svg", format="svg", dpi=300)
    return regrets


def rank_regret_trellis(dataset, save=False):
    """Prepares a trellis of all pairwise comparisons under two metrics - regret and rank.

    Parameters
    ----------
    dataset : str
        Dataset to analyze.
    save: bool
        Whether to save the resulting plot.

    Returns
    -------
    None
    """
    list_of_algs=["RFR", "RFC", "LRRF", "RPC"]

    fig, ax = plt.subplots(
        len(list_of_algs),
        len(list_of_algs),
        figsize=(7, 7),
        gridspec_kw={"wspace": 0.2, "hspace": 0.2},
    )
    rfr_df, rfc_df = load_result(dataset, "RFR", "RFC")
    lrrf_df, rpc_df = load_result(dataset, "LRRF", "RPC")
    all_dfs = [rfr_df, rfc_df, lrrf_df, rpc_df]
    # Getting max regret to set range for y axis
    all_diffs = []
    for i in range(rfc_df.shape[0]) :
        regrets = [rfr_df.iloc[i, 4], rfc_df.iloc[i, 4], lrrf_df.iloc[i, 4], rpc_df.iloc[i, 4]]
        for j, k in combinations(range(4), 2) :
            col_regret, row_regret = regrets[k], regrets[j]
            diff = row_regret - col_regret
            all_diffs.append(diff)
    min_regret_diff = min(all_diffs)
    max_regret_diff = max(all_diffs)

    vmax = (max_regret_diff // 10 + 1) * 10
    if min_regret_diff % 10 == 0 :
        vmin = (min_regret_diff // 10 - 1) * 10
    else :
        vmin = min_regret_diff // 10 * 10
    if max_regret_diff - min_regret_diff < 70 :
        jump = 10
    else :
        jump = 20
    
    yticks = np.arange(vmin, vmax+10, jump)
    yticklabels = np.arange(vmin, vmax+10, jump)
    # yticklabels = [x if x >= 0 else "{0}".format(u'\u2014') + f"{str(x)[1:]}" for x in yticklabels]
    
    # Making scatter plots of regrets
    # for row_df, col_df in combinations(all_dfs, 2):
    for i, j in combinations(range(4), 2):
        row_df = all_dfs[i]
        col_df = all_dfs[j]
        
        regrets = []
        for k in range(min(row_df.shape[0], col_df.shape[0])) :
            regret_vals = np.array([row_df.iloc[k,4], col_df.iloc[k,4]])
            if regret_vals[0] != regret_vals[1] :
                regrets.append(regret_vals[0] - regret_vals[1])
        regrets = np.sort(np.array(regrets))[::-1]

        # 5) Plotting regrets
        if i == 0 :
            showyticklabels = True
        else :
            showyticklabels = False
        ax[j, i].bar(np.arange(len(regrets)), regrets, width=0.9, color=["#5ec962"]*len(np.where(regrets>0)[0])+["#3b528b"]*len(np.where(regrets <0)[0]))
        ax[j, i].set_ylim(min_regret_diff-10, max_regret_diff+10)
        ax[j, i].set_yticks(yticks)
        xticks = np.arange(len(regrets))
        if len(regrets) <= 10 :
            xticklabels = np.arange(1, len(regrets)+1)
        else :
            xticklabels = []
            if len(regrets) <= 30 :
                for xval in xticks :
                    if xval % 3 == 0 :
                        xticklabels.append(xval+1)
                    else :
                        xticklabels.append("")
            else : 
                for xval in xticks :
                    if xval % 10 == 0 :
                        xticklabels.append(xval+1)
                    else :
                        xticklabels.append("")

        ax[j, i].set_xticks(xticks)
        ax[j, i].set_xticklabels(xticklabels, fontdict={"fontfamily":"arial", "fontsize":6})
        ax[j, i].axhline(0, 0, 1, color="k", linewidth=0.5)
        for axis in ["top", "bottom", "left", "right"]:
            ax[j, i].spines[axis].set_linewidth(1.5)
        if showyticklabels:
            ax[j, i].set_yticklabels(yticklabels, fontdict={"fontfamily":"arial", "fontsize":6})
        else:
            ax[j, i].set_yticklabels([])
        if j == 3 and i == 1 :
            ax[j, i].set_xlabel("Differently predicted substrates", fontdict={"fontfamily":"arial", "fontsize":6})
        if j == 3 and i == 0 :
            ax[j, i].legend(
            handles=[mpatches.Patch(color='#5ec962', label="Left"), mpatches.Patch(color="#3b528b", label="Top")],
            loc='lower left', fontsize=6, prop={"size":6, "family":"arial"}, title="Higher yield from\nmodel labeled on", title_fontproperties={"size":6, "family":"arial"})

        # rank heatmaps
        if j >= i + 2:
            showyticklabels = False
        counts = np.zeros((4,4)) # row : model1, col: model2
        bins = [0.33, 0.5, 1]
        for k in range(min(row_df.shape[0], col_df.shape[0])) :
            mrr_vals = np.array([row_df.iloc[k,2], col_df.iloc[k,2]])
            counts[np.digitize(mrr_vals, bins)[0], np.digitize(mrr_vals, bins)[1]] += 1

        # 3) Draw the heatmap
        sns.heatmap(counts, annot=True, square=True, cbar=False, annot_kws={"fontsize":6, "fontfamily":"arial"}, ax=ax[i, j], cmap="viridis")
        if j == i+1:
            ax[i, j].set_xticklabels(5-np.arange(1,5), fontdict={"fontfamily":"arial", "fontsize":6})
            ax[i, j].set_yticklabels(5-np.arange(1,5), fontdict={"fontfamily":"arial", "fontsize":6}, rotation=0)
        else :
            ax[i, j].tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

            
    # Empty diagonals
    for i in range(4) :
        for j in range(4) :
            if i == j : # Empty diagonals
                plt.setp(ax[j, i].spines.values(), visible=False)
                row_df = all_dfs[i]
                col_df = all_dfs[j]
                mrr = row_df["reciprocal_rank"].mean()
                ax[j, i].annotate("MRR = "+"{:.3f}".format(round(mrr, 3)), xy=(0.5, 0.5), xycoords="axes fraction", ha="center", fontfamily="arial", fontsize=10)
                if j == 0 :
                    ax[j, i].tick_params(colors="white")
                    ax[j, i].set_yticks(yticks)
                    ax[j, i].set_yticklabels(yticks, fontdict={"fontfamily":"arial", "fontsize":6, "color":"white"})
                    ax[j, i].set_ylabel(
                        list_of_algs[i], fontsize=10, fontfamily="arial", color="black"
                    )
                    ax[j, i].xaxis.set_label_position('top')
                    ax[j, i].set_xlabel(
                        list_of_algs[i], fontsize=10, fontfamily="arial", color="black"
                    )
                elif j == 3:
                    ax[j, i].tick_params(colors="white")
                    ax[j, i].yaxis.tick_right()
                    ax[j, i].set_yticks(yticks)
                    ax[j, i].set_yticklabels(yticks, fontdict={"fontfamily":"arial", "fontsize":6, "color":"white"})
                else :
                    ax[j, i].xaxis.set_visible(False)
                    ax[j, i].yaxis.set_visible(False)
                    
            elif i < j :
                if i == 0 :
                    ax[i, j].xaxis.set_label_position('top')
                    ax[i, j].set_xlabel(list_of_algs[j], fontsize=10, fontfamily="arial")    

            else :
                if j == 0:
                    ax[i, j].set_ylabel(list_of_algs[i], fontsize=10, fontfamily="arial")
    
    fig.suptitle(
        "Rank of selection",
        fontsize=10,
        x=0.613,
        y=0.93,
        fontweight="bold",
        fontfamily="arial",
    )
    fig.supylabel(
        "Yield(row model) - Yield(col model) (%)",
        fontsize=10,
        x=0.03,
        y=0.39,
        fontweight="bold",
        fontfamily="arial",
    )
    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.055, 0.055), 0.855, 0.855, fill=False, color="grey", lw=0.5, 
        zorder=1000, transform=fig.transFigure, figure=fig
    )
    fig.patches.extend([rect])
    if save: 
        plt.savefig(f"figures/SI/FigureS12_{dataset}_trellis.svg", dpi=300, format="svg")



### Borrowed from scikit_posthocs._plotting.py due to unknown error
from typing import Union, List, Tuple, Dict, Set
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


def plot_rr_heatmap(rr_array, model_list, dataset_list, hline_pos=[5,9], vline_pos=[1,2,4], square=False, cbar_kws=None, save=False):
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
    square : bool
        Whether to make the cells of the heatmap square.
    cbar_kws : None or dict
        Used to control the size of the colorbar.
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
    ax = sns.heatmap(ordered_rr_rank_by_dataset, cmap=cmap, annot=rr_array, square=square, cbar_kws=cbar_kws)
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
    
    average_ranks = rr_array.shape[1] + 1 - np.mean(ordered_rr_rank_by_dataset, axis=0)
    xticklabels = []
    for model_name, avg_rank in zip(model_list, average_ranks) :
        xticklabels.append(f"{model_name}\n({round(avg_rank, 1)})")
    ax.set_xticklabels(xticklabels, fontdict={"fontfamily":"arial", "fontsize":10})
    # ax.set_xticklabels(model_list, fontdict={"fontfamily":"arial", "fontsize":10})
    ax.set_yticklabels(dataset_list, fontdict={"fontfamily":"arial", "fontsize":10}, rotation=0)
    ax.set_xlabel("Models", fontdict={"fontfamily":"arial", "fontsize":12})
    ax.set_ylabel("Datasets", fontdict={"fontfamily":"arial", "fontsize":12})
    if type(save) == str:
        plt.savefig(f"figures/{save}", dpi=300, format="svg")


def plot_table_with_heatmap(rr_array, model_list, dataset_list, hline_pos=[5,9], last_hline=11, vline_pos=[1,2,4], square=False, cbar_kws=None, save=False, larger_is_better=True):
    """ Makes a table with heatmap."""
    ordered_rr_rank_by_dataset = rankdata(rr_array, axis=1)
    if larger_is_better :
        max_in_each_row = np.argmax(ordered_rr_rank_by_dataset, axis=1)
        second_in_each_row = np.argsort(ordered_rr_rank_by_dataset, axis=1)[:, -2]
        fmt = ".3f"
        average_ranks = [round(x,1) for x in rr_array.shape[1] + 1 - np.mean(ordered_rr_rank_by_dataset, axis=0)]
    else :
        max_in_each_row = np.argmin(ordered_rr_rank_by_dataset, axis=1)
        second_in_each_row = np.argsort(ordered_rr_rank_by_dataset, axis=1)[:, 1]
        fmt = ".1f"
        average_ranks = [round(x,1) for x in np.mean(ordered_rr_rank_by_dataset, axis=0)]
    array_to_plot = np.vstack((ordered_rr_rank_by_dataset, average_ranks))
    ax = sns.heatmap(
        array_to_plot,
        annot=np.vstack((rr_array, average_ranks)), 
        fmt=fmt,
        cbar=False, 
        cmap=ListedColormap(["white"]), 
        square=square,
        annot_kws={"fontfamily":"arial", "fontsize":10}
    )
    ncols=len(model_list)
    for row, col in enumerate(max_in_each_row) :
        ax.texts[ncols*row + col].set_weight("bold")
        ax.texts[ncols*row + col].set_color("green")
    for row, col in enumerate(second_in_each_row) :
        ax.texts[ncols*row + col].set_weight("bold")
    ax.texts[ncols*row+ncols+np.argmin(average_ranks)].set_weight("bold")
    ax.texts[ncols*row+ncols+np.argmin(average_ranks)].set_color("green")
    ax.texts[ncols*row+ncols+np.argsort(average_ranks)[1]].set_weight("bold")
    for t in ax.texts[ncols*row+ncols:] :
        t.set_text(str(t.get_text())[:3])
    
    xticklabels = []
    for model_name, avg_rank in zip(model_list, average_ranks) :
        xticklabels.append(f"{model_name}")
    yticklabels = deepcopy(dataset_list)
    yticklabels.append("Average rank\nacross datasets")
    ax.set_xticklabels(xticklabels, fontdict={"fontfamily":"arial", "fontsize":10})
    ax.set_yticklabels(yticklabels, fontdict={"fontfamily":"arial", "fontsize":10}, rotation=0)
    ax.set_xlabel("Models", fontdict={"fontfamily":"arial", "fontsize":12, "fontweight":"bold"})
    ax.set_ylabel("Datasets", fontdict={"fontfamily":"arial", "fontsize":12, "fontweight":"bold"})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    for v in vline_pos :
        ax.axvline(v,0,1, c="grey", lw=0.5)
    for h in hline_pos : 
        ax.axhline(h,0,1, c="grey", lw=0.5)
    ax.axhline(last_hline, 0, 1, c="black", lw=1.5)
    if type(save) == str:
        plt.savefig(f"figures/{save}", dpi=300, format="svg")


def plot_std_kde_plot(df_to_plot, model_order, h=0.3, aspect=10, xlim=(-0.05,0.2), xticks=[0, 0.05, 0.10, 0.15], save=False):
    """ Plots kde plots of how the standard deviations of MRR across different masks.
    
    Parameters
    ----------
    df_to_plot : pd.DataFrame object.
        Result dataframe to plot. Must have "std" as the name of column with std values.
    model_order : list of str.
        The order of models to use from top to bottom.
    h : float
        Height of the plot in inches
    aspect : float
        h*aspect is the width of the plot in inches
    xlim : tuple(float, float)
        The both ends of the x axis
    save : False or str
        if str, filename to save.
    """
    import warnings
    warnings.filterwarnings('ignore')
    pal = sns.cubehelix_palette(4, rot=-.25, light=.7)
    g = sns.FacetGrid(
        df_to_plot, row="model", hue="model", 
        aspect=aspect, height=h, palette=pal, row_order=model_order, xlim=xlim
    )

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "std",
        bw_adjust=.5, clip_on=False,
        fill=True, alpha=1, linewidth=0.5)
    g.map(sns.kdeplot, "std", clip_on=False, color="w", lw=0.5, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=0.5, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0, .4, label, color=color, #fontweight="bold", 
            ha="left", va="center", transform=ax.transAxes,
            fontname="arial", fontsize=10
        )

    g.map(label, "std", label="Standard deviation")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=.05)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xticks=xticks)
    g.despine(bottom=True, left=True)
    plt.xticks(fontname="arial", fontsize=8)
    plt.xlabel("Standard deviation", fontname="arial", fontsize=10)


    if type(save) == str:
        plt.savefig(f"figures/{save}", dpi=300, format="svg")