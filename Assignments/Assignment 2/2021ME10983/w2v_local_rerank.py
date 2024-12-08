import csv
import gzip
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize

import gensim
from gensim.models import Word2Vec


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')



# nltk.download('popular')

class DocumentStore:
    def __init__(self, doc_path, top100_file_path, queries_file_path):
        self.english_words = set(words.words())
        self.stop_words = set(stopwords.words('english'))
        self.doc_path = doc_path
        self.documents = {}
        self.collection_tokens = defaultdict(int)
        self.collection_length = 0
        self.query_top100 = defaultdict(list)
        self.query_id_map = {}
        self.all_doc_ids = set()
        # self.query_doc_rel = {}

        with open(top100_file_path, newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)
            for query_id, doc_id, score in reader:
                self.query_top100[query_id].append((doc_id, float(score)))
                self.all_doc_ids.add(doc_id)

        for query_id in self.query_top100:
            self.query_top100[query_id].sort(key=lambda x: x[1], reverse=True)
            self.query_top100[query_id] = [doc_id for doc_id, _ in self.query_top100[query_id]]

        with open(queries_file_path, newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)
            for query_id, query in reader:
                self.query_id_map[query_id] = query
        
        if not os.path.exists('extracted_docs.csv'):
            extracted_data = []
            with open(doc_path, 'r', encoding='utf-8') as f:
                cols = ['doc_id', 'url', 'title', 'body']
                i = 1
                for chunk in pd.read_csv(f, sep='\t', names=cols, chunksize=10000, skiprows=1):
                    filtered_chunk = chunk[chunk['doc_id'].isin(self.all_doc_ids)]
                    extracted_data.extend(filtered_chunk[['doc_id', 'title', 'body']].to_dict('records'))
                    # print(i)
                    i+=1
            output_df = pd.DataFrame(extracted_data)
            output_df.to_csv('extracted_docs.csv', index=False)


    def save_doc(self, row):
        st = time.time()
        doc_id, title, body = row.loc["doc_id"], row.loc["title"], row.loc["body"]
        if doc_id not in self.all_doc_ids: pass
        # print(doc_id, title, body)
        tokens = self.preprocess(title + ' ' + body)
        self.documents[doc_id] = tokens
        # self.collection_tokens.extend(tokens)
        for token in tokens:
            self.collection_tokens[token] += 1
        en = time.time()
        # print(f"time taken = {en-st}s")
        

        
    def load_and_preprocess(self):
        st = time.time()
        # print(os.path.exists('doc_store_documents1.pkl'))
        if os.path.exists('doc_store_documents.pkl'):
            with open('doc_store_documents.pkl', 'rb') as f:
                self.documents = pickle.load(f)
            with open('doc_store_collection_tokens.pkl', 'rb') as f:
                self.collection_tokens = pickle.load(f)
        else:
        # t = 0
            output_df = pd.read_csv('extracted_docs.csv')
            output_df.fillna(" ", inplace=True)

            output_df = output_df.apply(self.save_doc, axis=1)
            with open('doc_store_documents.pkl', 'wb') as f:
                pickle.dump(self.documents, f)

            with open('doc_store_collection_tokens.pkl', 'wb') as f:
                pickle.dump(self.collection_tokens, f)
        en = time.time()
        # print(f"preprocessing of top100 docs done in {en-st}")
        
        # self.collection_tokens = {token:freq for token, freq in self.collection_tokens.items() if freq >= 50}
        self.collection_length = len(self.collection_tokens)
    
    def preprocess(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words and token.isascii()]
        return tokens

class LanguageModel:
    def __init__(self, document_store, mu):
        self.document_store = document_store
        self.mu = mu
        
    def dirichlet_smoothed_score(self, query, doc_tokens):
        score = 0
        doc_length = len(doc_tokens)
        if type(query)==list:
            query_token = query
        else:
            query_token = self.document_store.preprocess(query)
        for term in query_token:
            # print(term, doc_tokens)
            term_freq_in_doc = doc_tokens.count(term)
            term_freq_in_collection = self.document_store.collection_tokens.get(term, 0)
            prob_term = (term_freq_in_doc + self.mu * (term_freq_in_collection / self.document_store.collection_length)) / (doc_length + self.mu)
            score += np.log(prob_term) if prob_term > 0 else 0
        return score

class RelevanceReranker:
    def __init__(self, document_store, language_model):
        self.document_store = document_store
        self.language_model = language_model
        self.query_reranked_docs = {}

        
    def rerank(self, query_id):
        query = self.document_store.query_id_map[query_id]
        scores = {}
        for doc_id in self.document_store.query_top100[query_id]:
            tokens = self.document_store.documents[doc_id]
            scores[doc_id] = self.language_model.dirichlet_smoothed_score(query, tokens)
            # print(f"reranking done for docid {doc_id}")
        reranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        self.query_reranked_docs[query_id] = reranked_docs
        return reranked_docs
    
    def save_reranked_docs_to_file(self, filename):
        with open(filename, 'w') as file:
            for query_id in self.document_store.query_id_map:
                reranked_docs = self.rerank(query_id)
                # print(f"reranking done for query_id {query_id}")
                # Sort the documents by score in descending order
                reranked_docs_sorted = sorted(reranked_docs, key=lambda x: x[1], reverse=True)
                # Write each document to file
                for rank, (doc_id, score) in enumerate(reranked_docs_sorted, start=1):
                    # Format: query_id Q0 doc_id rank score run_id
                    line = f"{query_id} Q0 {doc_id} {rank} {score:.2f} runid1\n"
                    file.write(line)

    # def metric_calculator(self, query_id, k):
    #     rel_list = []
    #     reranked_docs = self.query_reranked_docs[query_id]
    #     for doc_id, score in reranked_docs:
    #         rel = self.document_store.query_doc_rel.get((query_id, doc_id), 0)
    #         rel_list.append(int(rel))
    #     dcg = 0
    #     for i in range(1, k+1):
    #         dcg += rel_list[i-1]/np.log2(i+1)
    #     idcg = 0
    #     ideal_rel_list = sorted(rel_list, reverse=True)
    #     for i in range(1, k+1):
    #         idcg += ideal_rel_list[i-1]/np.log2(i+1)
    #     if idcg!=0: ndcg = dcg/idcg
    #     else: ndcg=0
    #     return ndcg

    # def calculate_all_metric(self, metric_file_path):
    #     metrics = {}
    #     for k in [5, 10, 50]:
    #         metrics[k] = {}
    #         for query_id in self.document_store.query_top100:
    #             metrics[k][query_id] = self.metric_calculator(query_id, k)
        
    #     df = pd.DataFrame.from_dict(metrics, orient='index')
    #     df.to_excel(metric_file_path)
        # print(f"Metrics successfully written to {metric_file_path}.")


####################### Main Task1##############################


class TextProcessor:
    """Process text for tokenization and cleaning."""
    def __init__(self, doc_store):
        # self.stop_words = set(stopwords.words('english'))
        self.doc_store = doc_store
    
    def preprocess(self, text):
        return self.doc_store.preprocess(text)

class EmbeddingModel:
    """Manages the Word2Vec embedding model."""
    def __init__(self, doc_store, vector_size=300, window=5, min_count=1, workers=4):
        self.doc_store = doc_store
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
    
    def train_model(self):
        tokenized_docs = list(self.doc_store.documents.values())
        tokenized_queries = [self.doc_store.preprocess(query) for query in self.doc_store.query_id_map.values()]
        for q in tokenized_queries: tokenized_docs.append(q)
        self.model = Word2Vec(sentences=tokenized_docs, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, workers=self.workers)
        
    def get_similar_words(self, word, top_n):
        if word in self.model.wv:
            return self.model.wv.most_similar(word, topn=top_n)
        return []

class QueryExpander:
    """Expands queries using a trained Word2Vec model."""
    def __init__(self, embedding_model, processor):
        self.embedding_model = embedding_model
        self.processor = processor
        self.U = None
        embedding_matrix = np.vstack([self.embedding_model.model.wv[word] for word in self.embedding_model.model.wv.index_to_key])
        self.U = normalize(embedding_matrix, norm='l2')

    def vectorize_query(self, query):
        vector = np.zeros(len(self.embedding_model.model.wv))
        indices = [self.embedding_model.model.wv.key_to_index[word] for word in query if word in self.embedding_model.model.wv.key_to_index]
        vector[indices] = 1
        return vector, indices
    
    def query_expansion(self, query_id, query, top_k, expansion_file_path):
        preprocess_query = self.processor.preprocess(query)
        q, ind = self.vectorize_query(preprocess_query)
        q = q.reshape(-1, 1)
        term_weights = self.U @ (self.U.T @ q)
        top_indices = np.argsort(term_weights, axis=0)
        top_indices = top_indices.flatten()
        expanded_terms = []
        i = 0
        while len(expanded_terms)!=top_k:
            if top_indices[i] in ind: 
                i+=1
                continue
            expanded_terms.append(self.embedding_model.model.wv.index_to_key[top_indices[i]])
            i+=1
        
        with open(expansion_file_path, 'a') as file:
            file.write(f'{query_id} : ')
            for term in expanded_terms:
                file.write(f"{term}, ")
            file.write("\n")

        expanded_terms.extend(preprocess_query)
        term_weights = term_weights.reshape(1, -1)
        return expanded_terms, term_weights[0]

class DocumentReranker:
    """Reranks documents based on expanded queries."""
    def __init__(self, query_expander, doc_store, language_model, reranker, n, lambda_):
        self.query_expander = query_expander
        self.doc_store = doc_store
        self.lm = language_model   
        self.reranker = reranker 
        self.n = n
        self.pq_plus = {}
        self.final_query_lang_model = {}
        self.lambda_ = lambda_
        self.query_reranked_docs = {}
        self.pq_dash = {}
        self.doc_lang_model = {}
        # self.new_queries = {}

    def scale_values_to_0_1(self, values):
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        scaled_values = [(value - min_val) / range_val for value in values]
        sm = 0
        for val in scaled_values: sm += val
        scaled_values = [val/sm for val in scaled_values]
        return scaled_values
    
    def make_new_queries(self, expansions_file_path):
        data = []
        for query_id in self.doc_store.query_top100:
            s = time.time()
            query = self.doc_store.query_id_map[query_id]
            new_query, term_weights = self.query_expander.query_expansion(query_id, query, self.n, expansions_file_path)
            # norm_term_weights = normalize(term_weights)
            # norm_term_weights = term_weights/sum(term_weights)
            norm_term_weights = self.scale_values_to_0_1(term_weights)
            self.pq_plus[query_id] = norm_term_weights
            # self.new_queries[query_id] = new_query
            new_query = " ".join(new_query)
            data.append({"query_id" : query_id, "text" : new_query})
            self.doc_store.query_id_map[query_id] = new_query
            e = time.time()
    
    def final_language_model(self):
        for query_id in self.doc_store.query_top100:
            query = self.doc_store.query_id_map[query_id]
            ls = []
            pq_dash = 0
            for w in self.query_expander.embedding_model.model.wv.key_to_index:
                pq = self.lm.dirichlet_smoothed_score(query, [w])
                final_prob = (1-self.lambda_)*np.exp(pq) + self.lambda_*self.pq_plus[query_id][self.query_expander.embedding_model.model.wv.key_to_index[w]]
                # if final_prob<=0:
                #     print(pq, self.pq_plus[query_id][self.query_expander.embedding_model.model.wv.key_to_index[w]])
                pq_dash += final_prob
                final_prob = np.log(final_prob)
                ls.append(final_prob)
            self.pq_dash[query_id] = pq_dash
            self.final_query_lang_model[query_id] = ls

    def doc_language_model(self):
        i = 1
        for doc_id in self.doc_store.documents:
            self.doc_lang_model[doc_id] = []
            tokens =  self.doc_store.documents[doc_id]
            for w in self.query_expander.embedding_model.model.wv.key_to_index:
                log_p = self.lm.dirichlet_smoothed_score([w], tokens)
                self.doc_lang_model[doc_id].append(log_p)
            # print(f"document language model created for doc number {i}")
            # print(self.doc_lang_model[doc_id])
            i+=1

    def rerank(self, query_id):
        query = self.doc_store.query_id_map[query_id]
        scores = {}        
        for doc_id in self.doc_store.query_top100[query_id]:
            tokens = self.doc_store.documents[doc_id]
            score = 0
            for token in tokens:
                score += self.final_query_lang_model[query_id][self.query_expander.embedding_model.model.wv.key_to_index[token]]
            scores[doc_id] = score
        reranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        self.query_reranked_docs[query_id] = reranked_docs
        return reranked_docs        


    def save_reranked_docs_to_file(self, filename):
        
        with open(filename, 'w') as file:
            for query_id in self.doc_store.query_id_map:
                reranked_docs = self.rerank(query_id)
                for rank, (doc_id, score) in enumerate(reranked_docs, start=1):
                    line = f"{query_id} Q0 {doc_id} {rank} {score:.5f} runid1\n"
                    file.write(line) 

    # def metric_calculator(self, query_id, k):
    #     rel_list = []
    #     reranked_docs = self.query_reranked_docs[query_id]
    #     for doc_id, score in reranked_docs:
    #         rel = self.doc_store.query_doc_rel.get((query_id, doc_id), 0)
    #         rel_list.append(int(rel))
        
    #     dcg = 0
    #     for i in range(1, k+1):
    #         dcg += rel_list[i-1]/np.log2(i+1)
    #     idcg = 0
    #     ideal_rel_list = sorted(rel_list, reverse=True)
    #     for i in range(1, k+1):
    #         idcg += ideal_rel_list[i-1]/np.log2(i+1)
        
    #     if idcg!=0: ndcg = dcg/idcg
    #     else: ndcg=0
    #     return ndcg

    # def calculate_all_metric(self, metric_file_path):
    #     metrics = {}
    #     for k in [5, 10, 50]:
    #         metrics[k] = {}
    #         for query_id in self.doc_store.query_top100:
    #             metrics[k][query_id] = self.metric_calculator(query_id, k)
        
    #     df = pd.DataFrame.from_dict(metrics, orient='index')
    #     df.to_excel(metric_file_path)
        # print(f"Metrics successfully written to {metric_file_path}.")


def main():
    query_file_path = sys.argv[1]
    query_top_100_path = sys.argv[2]
    docs_path = sys.argv[3]
    output_file_path = sys.argv[4]
    expansion_file_path = sys.argv[5]

    # document_file_path = "C:\\Users\\raj13\\docs.tsv"
    # top100_file_path = "top100docs.tsv"
    # queries_file_path = "queries.tsv"
    # qrels_file_path = "qrels.tsv"
    doc_store = DocumentStore(docs_path, query_top_100_path, query_file_path)
    doc_store.load_and_preprocess()
    # print("Document storing is done!!")
    processor = TextProcessor(doc_store)

    mu = 280
    lm = LanguageModel(doc_store, mu=mu)
    reranker = RelevanceReranker(doc_store, lm)
    # print("Reranker instantiated!!")

    embedding_model = EmbeddingModel(doc_store)
    embedding_model.train_model()
    query_expander = QueryExpander(embedding_model, processor)
    n = 10
    lambda_ = 0.8
    doc_reranker = DocumentReranker(query_expander, doc_store, lm, reranker, n, lambda_)
    doc_reranker.make_new_queries(expansion_file_path)

    doc_reranker.final_language_model()

    # output_filename = f"output_task1_{n}_major_upd.txt"
    # metrics_filename = f"metrics_task1_{n}_major_upd.xlsx"
    # reranker.save_reranked_docs_to_file(output_filename)
    # reranker.calculate_all_metric(metrics_filename)
    doc_reranker.save_reranked_docs_to_file(output_file_path)
    # doc_reranker.calculate_all_metric(metrics_filename)
    # print("Reranking done!!")
    print("Code Run Successful")

if __name__ == '__main__':
    main()

