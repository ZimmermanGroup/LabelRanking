import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from Dataloader import *
from Evaluator import *


def parse_args():
    parser = argparse.ArgumentParser(description="Specify the evaluation to run.")
    parser.add_argument(
        "--dataset",
        help="Which dataset to use. Should be either 'deoxy', 'natureHTE', 'informer'."
    )
    parser.add_argument(
        "--feature",
        help="Which feature to use. Should be either 'desc', 'fp', 'onehot', 'random'",
    )
    parser.add_argument(
        "--label_component",
        action="append",
        help="Which reaction components to consider as 'labels'.",
    )
    parser.add_argument(
        "--train_together",
        action=argparse.BooleanOptionalAction,
        help="Whether the non-label reaction component should be treated altogether or as separate datasets.",
    )
    parser.add_argument(
        "--rfr", action="store_true", help="Include Random Forest Regressor."
    )
    parser.add_argument(
        "--lrrf", action="store_true", help="Include Label Ranking RF as in Qiu, 2018"
    )
    parser.add_argument(
        "--lrt",
        action="store_true",
        help="Include Label Ranking Tree as in Hüllermeier, 2008",
    )
    parser.add_argument(
        "--rpc",
        action="store_true",
        help="Include Pairwise label ranking as in Hüllermeier, 2008",
    )
    parser.add_argument(
        "--ibm",
        action="store_true",
        help="Include Instance-based label ranking with Mallows model as in Hüllermeier, 2009",
    )
    parser.add_argument(
        "--ibpl",
        action="store_true",
        help="Include Instance-based label ranking with Plackett=Luce model as in Hüllermeier, 2010",
    )
    parser.add_argument(
        "--rfc",
        action="store_true",
        help="Include random forest classifier",
    ),
    parser.add_argument(
        "--lr", action="store_true", help="Include logistic regressor"
    ),
    parser.add_argument(
        "--knn", action="store_true", help="Include kNN classifier"
    ),
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include baseline models - avg_yield.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Whether to save resulting scores in an excel file.",
    )
    args = parser.parse_args()
    return args


def main():
    pass


if __name__ == "__main__" :
    parser = parse_args()
    if parser.dataset == "deoxy":
        if len(parser.label_component) == 2 :
            n_rxns = 4
            label_component = "both"
        else :
            n_rxns = 1
            label_component = parser.label_component[0]
        print(parser.train_together)
        data_reg = DeoxyDataset(True, label_component, parser.train_together, n_rxns)
        data = DeoxyDataset(False, label_component, parser.train_together, n_rxns)

        # rfr = RegressorEvaluator(
        #     data_reg,
        #     parser.feature,
        #     n_rxns,
        #     [GridSearchCV(
        #         RandomForestRegressor(random_state=42),
        #         param_grid={"n_estimators": [30, 100, 200], "max_depth": [5, 10, None]},
        #         scoring="r2",
        #         n_jobs=-1,
        #         cv=PredefinedSplit(np.repeat(np.arange(31), 20)),
        #     )],
        #     ["RFR"],
        #     PredefinedSplit(np.repeat(np.arange(32), 20))
        # ).train_and_evaluate_models()
        rfr = BaselineEvaluator(
            data,
            n_rxns,
            PredefinedSplit(np.repeat(np.arange(32), int(data.y_yield.shape[0] // 32)))
        )
        rfr.train_and_evaluate_models()
        print(rfr.perf_dict)
        perf_df = pd.DataFrame(rfr.perf_dict)
        # for model in perf_df["model"].unique():
            # print(
            #     model,
            #     round(
            #         perf_df[perf_df["model"] == model]["reciprocal_rank"].mean(), 3
            #     ),
            # )
        perf_df_by_sf = [perf_df.iloc[[x for x in range(perf_df.shape[0]) if x%5==i], :] for i in range(5)]
        for i in range(5):
            print(f"Sulfonyl fluoride {i+1}")
            sub_df = perf_df_by_sf[i]
            print(round(sub_df["reciprocal_rank"].mean(), 3))
            print()
        print()