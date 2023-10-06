import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scipy.stats import kendalltau, mstats
from label_ranking import *
from rank_aggregation import *
from sklr.tree import DecisionTreeLabelRanker
from tqdm import tqdm
import argparse

N_ITER = 5
N_CV = 10
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Select the models to evaluate.")
    parser.add_argument(
        "--algorithms", action="append", choices=["lrrf", "rpc", "ibm", "ibpl", "lrt"]
    )
    parser.add_argument(
        "--datasets", action="append", choices=["authorship", "elevators", "housing", "iris", "segment", "stock"]
    ) # num_instances = 841, 16599, 506, 150, 2310, 950, for each dataset respectively.
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Whether to save resulting scores in an excel file.",
    )
    parser.add_argument(
        "--missing_portion",
        default=0,
        type=float,
        help="Probability of each label missing a ranking."
    )
    args = parser.parse_args()
    return args

def main(parser):
    model_instances = {
        "lrrf":LabelRankingRandomForest(),
        "rpc":RPC(base_learner=LogisticRegression(C=1), cross_validator=None),
        "ibm":IBLR_M(n_neighbors=30, metric="euclidean"),
        "ibpl":IBLR_PL(n_neighbors=20),
        "lrt":DecisionTreeLabelRanker()
    }
    datasets = {
        "elevators":[9,9],
        "iris":[4,3],
        "segment":[18,7],
        "housing":[6,6],
        "authorship":[70,4],
        "stock":[5,5]
    }
    score_dict = {
        model_name:np.zeros((len(parser.datasets), N_ITER * N_CV)) for model_name in parser.algorithms
    }

    for i, dataset_name in enumerate(parser.datasets) :
        dataset_df = pd.read_excel(
            f"datasets/benchmarks/{dataset_name}_dense.xls", header=None
        )
        X = dataset_df.iloc[:, : datasets[dataset_name][0]].to_numpy()
        y = dataset_df.iloc[:, -1 * datasets[dataset_name][1] :].to_numpy()
        for j in tqdm(range(N_ITER)):
            kf = KFold(n_splits=N_CV, shuffle=True, random_state=42 + j)

            for k, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if parser.missing_portion > 0 :
                    y_train_actual = deepcopy(y_train)
                    random_array = np.random.rand(y_train.shape[0], y_test.shape[1])
                    y_train_missing = deepcopy(y_train).astype(float)
                    y_train_missing[np.where(random_array <= parser.missing_portion)] = np.nan
                    # All three shouldn't be nans
                    for row_ind in np.where(np.sum(np.isnan(y_train_missing), axis=1)==y_train_missing.shape[1]) :
                        y_train_missing[row_ind, row_ind%y_train_missing.shape[1]] = y_train_actual[row_ind, row_ind%y_train_missing.shape[1]]
                    y_train = mstats.rankdata(np.ma.masked_invalid(y_train_missing), axis=1)
                    y_train[y_train == 0] = np.nan
                    
                for l, model_name in enumerate(parser.algorithms):
                    model = model_instances[model_name]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score_dict[model_name][i, N_CV * j + k] = kendalltau(
                        y_pred, y_test
                    ).statistic
    average_scores = np.zeros((len(parser.algorithms), len(parser.datasets)))
    for k, v in score_dict.items():
        average_scores[parser.algorithms.index(k), :] = np.mean(v, axis=1).flatten()
    score_df = pd.DataFrame(
        data=average_scores, index=parser.algorithms, columns=parser.datasets
    )
    print(score_df)
    if parser.save:
        score_df.to_excel("performance_excels/benchmarks/Benchmark_scores.xlsx")

if __name__ == "__main__":
    parser = parse_args()
    main(parser)