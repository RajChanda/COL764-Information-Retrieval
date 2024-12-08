import struct
import json
import re
from collections import defaultdict, Counter
import math
import sys


class TF_IDF_Search:
    def __init__(self):
        self.merges_BPE = {}
        self.vocabulary_WordPiece = set()
        self.tokenizer_choice = None
        self.num_docs = 192509
        # self.num_docs = 100
        self.vocabulary = {}
        self.postings = defaultdict(dict)
        self.doc_map = {}
        self.query_mapping = {}
        self.query_tokens = defaultdict(list)
        self.tokenized_queries = []
        self.precomputed_weight_doc = []
        self.precomputed_weight_query = []
        self.rankings = []
        self.results = []
    
    def get_tokenizer_choice(self):
        with open("tokenizer_choice_file.txt", "r") as file:
            self.tokenizer_choice = int(file.readline().strip())

    def get_tokenizer_data(self):
        if self.tokenizer_choice==1:
            return
        elif self.tokenizer_choice==2:
            with open("merges_BPE.txt", 'r') as file:
                for line in file:
                    line = line.strip()
                    key, value = line.split(':')
                    key1, key2 = key.split(',')
                    self.merges_BPE[(key1, key2)] = value
        elif self.tokenizer_choice==3:
            with open("vocabulary_WordPiece.txt", "r") as file:
                for line in file:
                    word = line.strip()
                    self.vocabulary_WordPiece.add(word)
        return
    
    def encode_SimpleTokenizer(self, text):
        text = re.sub(r'[^a-zA-Z\s]+', ' ', text)
        tokens = text.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].encode('ascii', errors='ignore').decode('ascii').lower()
        tokens = [token for token in tokens if token]
        for token in tokens:
            token = self.vocabulary[token]
        return tokens

    def encode_BPE(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        upd_text = text.split()
        for i in range(len(upd_text)):
            upd_text[i] = upd_text[i] + "|"
        tokenized_text = []
        for word in upd_text:
            if word in self.query_tokens:
                # print(word, self.query_tokens[word])
                tokenized_text.extend(self.query_tokens[word])
            else:
                split = [l for l in word]
                for pair, merge in self.merges_BPE.items():
                    i = 0
                    while i < len(split) - 1:
                        if split[i] == pair[0] and split[i + 1] == pair[1]:
                            split = split[:i] + [merge] + split[i + 2 :]
                        else:
                            i += 1
                    # splits[idx] = split
                tokenized_text.extend(split)
                self.query_tokens[word] = split
        for token in tokenized_text:
            token = self.vocabulary[token]
        return tokenized_text
    
    def encode_WordPiece(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        words = text.split()
        encoded_words = []
        for word in words:
            if word in self.query_tokens:
                encoded_words = self.query_tokens[word]
                encoded_words.extend(encoded_words)
            else:                
                tokens = []
                while len(word) > 0:
                    i = len(word)
                    while i > 0 and word[:i] not in self.vocabulary_WordPiece:
                        i -= 1
                    if i == 0:
                        return ["[OOD]"]
                    tokens.append(word[:i])
                    word = word[i:]
                    if len(word) > 0:
                        word = f"|{word}"
                encoded_words.extend(tokens)
        for token in encoded_words:
            token = self.vocabulary[token]
        # print(encoded_words)
        return encoded_words
    
    def get_tokens(self, text):
        if self.tokenizer_choice==1:
            return self.encode_SimpleTokenizer(text)
        elif self.tokenizer_choice==2:
            return self.encode_BPE(text)
        elif self.tokenizer_choice==3:
            return self.encode_WordPiece(text)

    def read_vocab(self, vocab_file):
        with open(vocab_file, 'r') as vf:
            for line in vf:
                # print(line.strip().split(", "))
                (token_id, token) = line.strip().split(", ")
                self.vocabulary[token] = int(token_id)

    def read_index(self, index_file):
        with open(index_file, 'rb') as pf:
            while True:
                token_id_data = pf.read(4)
                if not token_id_data:
                    break
                token_id = struct.unpack('I', token_id_data)[0]
                num_postings_data = pf.read(4)
                num_postings = struct.unpack('I', num_postings_data)[0]
                postings_list = {}
                for _ in range(num_postings):
                    doc_id_data = pf.read(4)
                    doc_id = struct.unpack('I', doc_id_data)[0]
                    num_positions_data = pf.read(4)
                    num_positions = struct.unpack('I', num_positions_data)[0]
                    postings_list[int(doc_id)] = int(num_positions)
                self.postings[int(token_id)] = postings_list

    def get_postings(self, token):
        token_id = self.vocabulary.get(token, None)
        if token_id is None:
            return None
        return self.postings.get(token_id, [])
    
    def get_doc_mapping(self, doc_file_path):
        with open(doc_file_path, "r") as file:
            for line in file:
                text = line.strip()
                doc_num, doc_id = text.split(":")
                self.doc_map[int(doc_num)] = doc_id


    def get_queries(self, query_file):

        with open(query_file, "r") as file:
            for i, line in enumerate(file):
                article = json.loads(line.strip())
                q_id = article.get("query_id")
                title = article.get("title", "")
                desc = article.get("description", "")
                content = title+ " " + desc
                tokens = self.get_tokens(content)
                count_dict = Counter(tokens)
                query_dict = {}
                temp = 0
                for token, f in count_dict.items():
                    val = (1 + math.log(f, 2)) * math.log(self.num_docs + 1)
                    query_dict[self.vocabulary[token]] = val
                    temp += val*val
                temp = math.sqrt(temp)
                self.precomputed_weight_query.append([query_dict, temp])
                self.tokenized_queries.append(tokens)
                self.query_mapping[i+1] = q_id


    def get_tf(self, token, doc):
        f = self.postings[token].get(doc, 0)
        if f==0: return 1
        tf = 1 + math.log(f, 2)
        return tf
    
    def get_idf(self, token):
        df = len(self.postings[token])
        idf = math.log(1 + (self.num_docs/df), 2)
        return idf
    
    def get_weight(self, token, doc):
        return self.get_tf(token, doc) * self.get_idf(token)

    def precompute(self):
        V = len(self.vocabulary)
        N = self.num_docs
        # print(V, N)
        i = 0
        for doc in range(1, N+1):
            w = 0
            for token in range(1, V+1):
                val = self.get_weight(token, doc)
                w += val**2
                i+=1
            self.precomputed_weight_doc.append(math.sqrt(w))
            # if i%1e3==0: print(f"{i} iterations completed")
    
    def get_similarity(self, query_id, doc):
        precomputed_query_weights = self.precomputed_weight_query[query_id-1][0]
        num = 0
        for key, value in precomputed_query_weights.items():
            w_ij = self.get_weight(key, doc)
            num += w_ij * value        
        den = self.precomputed_weight_doc[doc-1] * self.precomputed_weight_query[query_id-1][1]
        return num/den
    
    def get_rankings(self):
        for i in range(1, len(self.tokenized_queries)+1):
            ranks = {}
            for j in range(1, self.num_docs+1):
                cos_similarity = self.get_similarity(i, j)
                ranks[self.doc_map[j]] = cos_similarity
            sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
            self.rankings.append(sorted_ranks)
            
            # print(f"ranking achieved for query : {i}")
        


    def get_performance(self, qrels_file, x):
        query_relevant = defaultdict(set)
        with open(qrels_file, "r") as file:
            for line in file:
                article = json.loads(line.strip())

                query_id = article.get("query_id", "")
                doc_id = article.get("doc_id", "")
                relevance = article.get("relevance", "")
                if relevance!=0 : query_relevant[query_id].add(doc_id)
        
        for i, (q_id, A) in enumerate(query_relevant.items()):
            retrieved = self.rankings[i]
            B = set()
            for j, val in enumerate(retrieved):
                # print(val)
                if j == x: break
                B.add(val[0])
                j+=1
            num = A & B
            precision = float(len(num))/float(x)
            recall = float(len(num))/float(len(A))
            if precision==0:
                F = 0
            else:
                F = 2 * precision * recall / (precision + recall)
            print(f"metrics for query {q_id} -> precision : {precision}, recall : {recall}, F{x} : {F}")

    def write_results(self, results_file):
        with open(results_file, "w") as file:
            file.write("qid\titeration\tdocid\trelevancy\n")
            for i in range(len(self.rankings)):
                qid = self.query_mapping[i+1]
                iteration = 0
                j = 0
                for doc_id, relevance in self.rankings[i]:
                    if j==100: break
                    file.write(f"{qid}\t{iteration}\t{doc_id}\t{relevance}\n")
                    j+=1

    def run_tfidf(self, vocab_file, index_file, query_file):

        self.read_vocab(vocab_file)
        self.read_index(index_file)
        print("dictionary and postings are retrieved")
        self.get_doc_mapping("doc_map.txt")
        print("document mapping retreived")
        self.get_tokenizer_choice()
        self.get_tokenizer_data()
        print("tokenizer data retreived")
        self.get_queries(query_file)
        print("queries are tokenized and query weights are calculated")
        self.precompute()
        print("document weights are calculated")
        self.get_rankings()
        print("rankings are done")


        # self.get_performance("train_data\\cord19-trec_covid-qrels.txt", x=100)


query_file = sys.argv[1]
results_file = sys.argv[2]
index_file = sys.argv[3]
vocab_file = sys.argv[4]

    
# vocab_file = "index.dict"
# index_file = "index.idx"
# query_file = "train_data\\cord19-trec_covid-queries.txt"
# results_file = "results_file.txt"

# doc_file = "train_data\\cord19-trec_covid-docs.txt"
# doc_file = "train_data\\rough.txt"

# start_time = time.time()

tfidf = TF_IDF_Search()
tfidf.run_tfidf(vocab_file, index_file, query_file)
tfidf.write_results(results_file)

# print(f"total time for ranking retreival : {time.time() - start_time}")

# print(tfidf.query_tokens)
# print(tfidf.tokenized_queries)
# print(tfidf.rankings)




