import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr
from label_ranking import *
from rank_aggregation import *
from tqdm import tqdm
import joblib
import argparse

DATASET_INFO = {
    "name":["iris"], # ,"authorship",  "elevators", "housing", "segment", "stock"
    "num_features":[4], # ,70,9,6,18,5
    "num_labels":[3] # ,4,9,6,7,5
}
N_ITER = 5
N_CV = 10

lrrf_tau_vals = np.zeros((len(DATASET_INFO["name"]), N_ITER*N_CV))
rpc_tau_vals = np.zeros((len(DATASET_INFO["name"]), N_ITER*N_CV))

def parse_args():
    parser = argparse.ArgumentParser(description="Select the models to evaluate.")
    parser.add_argument("--lrrf", default=False, help="Include Label Ranking RF as in Qiu, 2018")
    parser.add_argument("--rpc", default=False, help="Include Pairwise label ranking as in Hüllermeier, 2008")
    parser.add_argument("--ibm", default=False, help="Include Instance-based label ranking with Mallows model as in Hüllermeier, 2009")
    parser.add_argument("--save", default=True, help="Whether to save resulting scores in an excel file.")
    args = parser.parse_args()
    if args.lrrf is False and args.rpc is False and args.ibm is False :
        parser.error("At least one model is required.")
    return args


if __name__ == "__main__" :
    parser = parse_args()
    datasets_info_df = pd.DataFrame(DATASET_INFO)

    score_dict = {}
    model_names = []
    models = []
    
    if parser.lrrf :
        model_names.append("Label Ranking Random Forest")
        models.append(LabelRankingRandomForest())
        score_dict.update(
            {"Label Ranking Random Forest":np.zeros((len(DATASET_INFO["name"]), N_ITER * N_CV))}
        ) 
    if parser.rpc :
        model_names.append("Pairwise Comparison") 
        models.append(
            RPC(
                base_learner=LogisticRegression(C=1),
                cross_validator=None
            )
        )
        score_dict.update(
            {"Pairwise Comparison":np.zeros((len(DATASET_INFO["name"]), N_ITER * N_CV))}
        ) 
    if parser.ibm :
        model_names.append("Instance Based Mallows") 
        models.append(IBLR_M(n_neighbors=5, metric="euclidean"))
        score_dict.update(
            {"Instance Based Mallows":np.zeros((len(DATASET_INFO["name"]), N_ITER * N_CV))}
        ) 


    for i, row in datasets_info_df.iterrows() :
        dataset = pd.read_excel(f"datasets/{row['name']}_dense.xls", header=None)
        X = dataset.iloc[:,:row["num_features"]].to_numpy()
        y = dataset.iloc[:,-1*row["num_labels"]:].to_numpy()
        
        for j in tqdm(range(N_ITER)) :
            kf = KFold(n_splits=N_CV, shuffle=True, random_state=42+j)
            
            for k, (train_index, test_index) in enumerate(kf.split(X)) :
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                for i, model in enumerate(models) :
                    model.fit(X_train, y_train) 
                    y_pred = model.predict(X_test)
                    score_dict[model_names[i]][i, N_CV*j+k] = kendalltau(len(y_pred)-y_pred, y_test).statistic
    print(score_dict)
    average_scores = np.zeros((len(model_names), len(DATASET_INFO["name"])))
    for k, v in score_dict.items() :
        average_scores[model_names.index(k), :] = np.mean(v, axis=1).flatten()
    score_df = pd.DataFrame(data=average_scores, index=model_names, columns=DATASET_INFO["name"])
    if parser.save :
        score_df.to_excel("Benchmark_scores.xlsx")
    
    print(score_df)
        