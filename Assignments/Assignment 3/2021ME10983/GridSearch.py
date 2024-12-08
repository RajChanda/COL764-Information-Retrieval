import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import itertools
from sklearn.preprocessing import StandardScaler


class myGridSearch:
    def __init__(self, params, year, dataset):
        self.params = params
        self.num_params = len(params)
        self.year = year
        self.model_dict = {"SVR" : SVR, 'GBR' : GradientBoostingRegressor}
        self.dataset = dataset
        
        cols = ["Model", "Fold", "Loss", "Params"]
        # cols.extend(params.keys())
        self.cv_results = pd.DataFrame(columns=cols)
        self.agg_res = pd.DataFrame(columns=["Model", "Params", "Avg Loss"])
    
    def normalize(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def dcg(self, relevances, k=10):
        relevances = np.asfarray(relevances)[:k]
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))

    def ndcg(self, relevances, k=10):
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self.dcg(ideal_relevances, k)
        if idcg == 0:
            return 0  
        return self.dcg(relevances, k) / idcg

    def start_search(self, model_name, k):
        model = self.model_dict[model_name]
        for params_collection in self.params:
            param_combinations = list(itertools.product(*(params_collection[name] for name in params_collection)))
            
            for combination in param_combinations:
                param_dict = dict(zip(params_collection.keys(), combination))
                print("Running model with parameters:", param_dict)
                loss_ls = 1
                for fold in [1, 2, 3, 4]:
                    print(f"Running model with parameters: {param_dict} on fold {fold}")
                    df1 = self.dataset[self.year][fold]['training']
                    df2 = self.dataset[self.year][fold]['validation']

                    X_train = df1.drop(columns=["label", "qid", "doc_id"])
                    y_train = df1["label"]

                    X_val = df2.drop(columns=["label", "qid", "doc_id"])
                    y_val = df2["label"]
                    
                    X_train, X_val = self.normalize(X_train, X_val)
                    
                    regressor = model(**param_dict).fit(X_train, y_train)
                    print("fitting done")
                    y_pred = regressor.predict(X_val)

                    results_df = pd.DataFrame({'query_id' : df2['qid'],
                                            'doc_id' : df2['doc_id'],
                                            'predicted_relevance' : y_pred,
                                            'actual_relevance' : y_val})
                    
                    results_df.sort_values(by=['query_id', 'predicted_relevance'], ascending=[True, False], inplace=True)

                    ndcg_scores = results_df.groupby('query_id').apply(
                        lambda x: self.ndcg(x['actual_relevance'], k=k)
                    )                
                    average_ndcg = ndcg_scores.mean()             
                    print(f"average nDCG_{k} : {average_ndcg}")


                    # loss = self.get_loss(y_val, y_pred)
                    print(f"loss : {average_ndcg}")
                    ls = [model, fold, average_ndcg, param_dict]
                    self.cv_results.loc[len(self.cv_results)] = ls
                    loss_ls = min(loss_ls, average_ndcg)
                
                ls = [model, param_dict, loss_ls]
                self.agg_res[len(self.agg_res)] = ls
                print(f"nDCG for params : {param_dict} is {loss_ls}")

