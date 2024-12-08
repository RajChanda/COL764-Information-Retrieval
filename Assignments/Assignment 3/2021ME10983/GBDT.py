import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import itertools
from sklearn.preprocessing import StandardScaler
import os
import sys
import yaml

def read_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            elements = line.strip().split()
            label = int(elements[0])
            qid = elements[1].split(':')[1]
            features = {}
            doc_id = None
            for elem in elements[2: ]:
                if elem.startswith('#docid'):
                    doc_id = elements[-1]
                    break
                else:
                    feature_id, value = elem.split(':')
                    features[str(feature_id)] = float(value)
            data.append({'label': label, 'qid': qid, 'features': features, 'doc_id': doc_id})
    df = pd.DataFrame(data)
    features_df = pd.DataFrame(df['features'].tolist())
    df_expanded = pd.concat([df, features_df], axis=1)
    df_expanded.drop('features', axis=1, inplace=True)
    
    return df_expanded


class get_metrics:
    def __init__(self, model, params, dataset):
        self.model = model
        self.data = dataset
        self.params = params
        self.all_fold_results = None
    

    def normalize(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def dcg(self, relevances, k=10):
        # """Discounted cumulative gain at rank k for binary relevance."""
        relevances = np.asfarray(relevances)[:k]
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))

    def ndcg(self, relevances, k=10):
        # """Normalized discounted cumulative gain at rank k for binary relevance."""
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self.dcg(ideal_relevances, k)
        if idcg == 0:
            return 0  # Handle the case where there are no relevant documents
        return self.dcg(relevances, k) / idcg
    
    def get_results(self, year):
        # self.results_df = pd.DataFrame(columns=['query_id', 'predicted_relevance', 'actual_relevance'])
        
        for fold in [5]:
            test = self.data[year][fold]['test']
            train = self.data[year][fold]['training']
            
            actual_relevance = test['label']
            norm_train, norm_test = self.normalize(train.drop(columns=['qid', 'doc_id', 'label']), test.drop(columns=['qid', 'doc_id', 'label']))

            regressor = self.model(**self.params).fit(norm_train, train['label'])

            predicted_relevance = regressor.predict(norm_test)

            results_df = pd.DataFrame({'query_id' : self.data[year][fold]['test']['qid'],
                                       'doc_id' : self.data[year][fold]['test']['doc_id'],
                                       'predicted_relevance' : predicted_relevance,
                                       'actual_relevance' : actual_relevance})
            
            results_df.sort_values(by=['query_id', 'predicted_relevance'], ascending=[True, False], inplace=True)

            self.all_fold_results = pd.concat([self.all_fold_results,results_df], axis=0)

            # ndcg_scores = results_df.groupby('query_id').apply(
            #     lambda x: self.ndcg(x['actual_relevance'], k=k)
            # )

            # average_ndcg = ndcg_scores.mean()
            # print(f"Average nDCG_10 for fold {fold}:", average_ndcg)
            # print(ndcg_scores)
            # return average_ndcg

    def save_results_to_file(self, output_file_path):
        formatted_df = self.all_fold_results[['query_id', 'doc_id', 'predicted_relevance']]
        formatted_df.columns = ['qid', 'docid', 'relevancy']
        formatted_df.insert(1, 'iteration', 0)
        # formatted_df.dropna(inplace=True)
        formatted_df.sort_values(by=['qid', 'relevancy'], ascending=[True, False], inplace=True)
        with open(output_file_path, 'w') as file:
            file.write("qid\titeration\tdocid\trelevancy\n")
            for index, row in formatted_df.iterrows():
                file.write(f"{row['qid']}\t{row['iteration']}\t{row['docid']}\t{row['relevancy']}\n")

        # print(f"Results saved to {output_file_path}")    

def main():            
    fold_directory = sys.argv[1]
    output_file_path = sys.argv[2]
    params_file_path = sys.argv[3]
                
    with open(params_file_path, "r") as file:
        params = yaml.safe_load(file)

    # print(params)

    year = None
    if params_file_path=="td2003-gbp.yaml": year = 2003
    else: year = 2004
                
    complete_dataset = {}
    for yr in [year]:
        complete_dataset[yr] = {}
        for fold in [5]:
            complete_dataset[yr][fold] = {}
            for ds in ['training', 'test']:
                file_path = f"{fold_directory}\\{ds}set.txt"
                complete_dataset[yr][fold][ds] = read_dataset(file_path)


    model = GradientBoostingRegressor
    metrics = get_metrics(model, params, complete_dataset)
    metrics.get_results(year=year)
    metrics.save_results_to_file(output_file_path=output_file_path)



if __name__ == '__main__':
    main()



