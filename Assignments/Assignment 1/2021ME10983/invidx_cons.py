import struct
import json
import time
import re
from collections import defaultdict
import sys


class SimpleTokenizer:
    def __init__(self):
        self.vocabulary = defaultdict(int)

    def tokenizer(self, text):
        # pattern = r'[ ,.:;"\']'
        # # pattern = r"[ ,.:;\"'()\[\]]+"
        # tokens = re.split(pattern, text)
        text = re.sub(r'[^a-zA-Z\s]+', ' ', text)
        tokens = text.split()
        tokens = [token.lower() for token in tokens if token]
        for token in tokens:
            self.vocabulary[token] += 1

    def process_articles(self, file_path, fields=["title", "abstract"]):
        with open(file_path, 'r') as file:
            for line in file:
                article = json.loads(line.strip())
                
                for field in fields:
                    field_text = article.get(field, "")                
                    self.tokenizer(field_text)
    
    def get_tokens(self, text):
        text = re.sub(r'[^a-zA-Z\s]+', ' ', text)
        tokens = text.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].encode('ascii', errors='ignore').decode('ascii').lower()
        tokens = [token for token in tokens if token]
        return tokens
    
    def train_tokenizer(self, file_path):
        self.process_articles(file_path)

    def write(self, vocab_file_path):
        with open(vocab_file_path, "w") as file:
            for token in self.vocabulary:
                file.write(token + "\n")

        # print(f"Tokens have been saved to {vocab_file_path}")


class BPE:
    def __init__(self):
        self.vocabulary = []
        self.corpus = defaultdict(list)
        self.merges = {}
        self.encoded_merges = {}
        self.vocab1 = {idx: bytes([idx]) for idx in range(256)}

        # self.temp = defaultdict(list)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        for i in range(len(words)):
            words[i] = words[i] + "|"
        return words
    
    def get_initial_corpus(self, file_path):
        initial_vocab = set()   
        # initial_corpus = defaultdict(int)
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i >= 96255:
                    break
                article = json.loads(line.strip())
                title = article.get("title", "")
                abstract = article.get("abstract", "")
                for text in [title, abstract]:
                    if "|" in text:
                        continue
                    words = self.preprocess_text(text)
                    for element in words:
                        word = element
                        # word = word+"|"
                        word = word.encode('ascii', errors='ignore').decode('ascii').encode('utf-8')
                        # print(element)
                        letter_list = list(map(int, word))
                        # print(letter_list)
                        # letter_tuple = tuple(letter_list)
                        # initial_corpus[letter_tuple] += 1
                        if element in self.corpus:
                            ls = self.corpus[element]
                            # print(ls)
                            ls[-1] += 1
                            self.corpus[element] = ls
                            # print(self.corpus[element])
                        else:
                            ls = letter_list
                            ls.append(1)
                            self.corpus[element] = ls
                        initial_vocab.update(word)

                        # self.temp[element] = letter_list
        
        self.vocabulary = sorted(initial_vocab)
        # print(self.vocabulary)
        # print(len(self.temp))
        # for key, val in initial_corpus.items():
        #     ids = list(key)
        #     ids.append(val)
        #     self.corpus.append(ids)
        # del initial_corpus
        # print(len(self.corpus))
        # print(self.corpus)

    def get_pair_freq(self):
        counts = {}
        for key, value in self.corpus.items():
            ids = value[:-1]
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + ids[-1]

        top_pair = max(counts, key=counts.get)
        return counts, top_pair    

    def merge_op(self, top_pair, idx):
        for keys, values in self.corpus.items():
            word = keys
            line = values[:-1]
            freq = values[-1]
            l = len(line)
            i = 0
            new_line = []
            flag = 0
            while i < l:
                if i < l-1 and line[i]==top_pair[0] and line[i+1]==top_pair[1]:
                    new_line.append(idx)
                    i+=2
                    flag=1
                else:
                    new_line.append(line[i])
                    i+=1
            new_line.append(freq)
            if flag==1:
                self.corpus.update({word : new_line})
            # if flag==1:
            #     print(top_pair, idx)
            #     print(line, freq)
            #     print(new_line)

        # vocab.add(idx)
        self.vocabulary.append(idx)
    
    def make_all_merges(self, desired_vocab_size, verbose):
        num_merges = desired_vocab_size - len(self.vocabulary)
        for i in range(num_merges):
            start = time.time()
            counts, top_pair = self.get_pair_freq()
            idx = 256 + i
            if verbose==1 : 
                if i%100==0: print(f"{i+1} merges done")
                # print(top_pair)
            self.merge_op(top_pair, idx)
            self.encoded_merges[top_pair] = idx
            # print(time.time()-start)

    def decode(self, ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab1[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        # given a string, return list of integers (the tokens)
        words = self.preprocess_text(text)
        encoded_text = []
        for word in words:
            # if word in self.corpus:
            #     encoded_text.append(self.corpus[word][:-1])
            #     continue
            # word = word + "|"
            word = word.encode('ascii', errors='ignore').decode('ascii').encode('utf-8')
            tokens = list(map(int, word))
            while len(tokens) >= 2:
                counts = {}
                for pair in zip(tokens, tokens[1:]):
                    counts[pair] = counts.get(pair, 0)+1
                pair = min(counts, key=lambda p: self.encoded_merges.get(p, float("inf")))
                if pair not in self.encoded_merges:
                    break # nothing else can be merged
                idx = self.encoded_merges[pair]
                i = 0
                upd_text = []
                while i < len(tokens):
                    if i < len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]:
                        upd_text.append(idx)
                        i+=2
                    else:
                        upd_text.append(tokens[i])
                        i+=1
                tokens = upd_text
            for elements in tokens:
                encoded_text.append(elements)
        return encoded_text
    
    def train_tokenizer(self, file_path, desired_vocab_size, verbose):
        self.get_initial_corpus(file_path)
        self.make_all_merges(desired_vocab_size, verbose)
        
        self.vocab1 = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.encoded_merges.items():
            self.vocab1[idx] = self.vocab1[p0] + self.vocab1[p1]
        
        for i in range(len(self.vocabulary)):
            self.vocabulary[i] = self.decode([self.vocabulary[i]])

        for (token1, token2), token3 in self.encoded_merges.items():
            decoded_token1 = self.decode([token1])
            decoded_token2 = self.decode([token2])
            decoded_token3 = self.decode([token3])
            self.merges[tuple([decoded_token1, decoded_token2])] = decoded_token3
        # print(self.corpus)
        for key, value in self.corpus.items():
            ls = []
            for elem in value[:-1]:
                ls.append(self.decode([elem]))
            ls.append(value[-1])
            self.corpus.update({key:ls})

    def write_vocabulary(self, vocab_file_path):
        with open(vocab_file_path, "w") as file:
            for token in self.vocabulary:
                file.write(token + "\n")

        # print(f"Tokens have been saved to {vocab_file_path}")

    def write_merges(self, merge_file_path):
        with open(merge_file_path, "w") as file:
                for key, value in self.merges.items():
                    file.write(f"{key[0]},{key[1]}:{value}\n")
        # print(f"Merges have been saved to {merge_file_path}")

    def get_tokens(self, text):
        tokenized_text = []
        upd_text = self.preprocess_text(text)
        upd_text = [word for word in upd_text]
        # splits = [[l for l in word] for word in upd_text]
        # for split in splits:
        #     split.append("|")
            
        # print(splits)
        tokenized_text = []
        for word in upd_text:
            if word in self.corpus:
                tokenized_text.extend(self.corpus[word][:-1])
                continue
            else:
                split = [l for l in word]
                for pair, merge in self.merges.items():
                    i = 0
                    while i < len(split) - 1:
                        if split[i] == pair[0] and split[i + 1] == pair[1]:
                            split = split[:i] + [merge] + split[i + 2 :]
                        else:
                            i += 1
                    # splits[idx] = split
                tokenized_text.extend(split)
        return tokenized_text
    


class WordPiece:
    def __init__(self):
        self.word_split = defaultdict(int)
        self.word_freq = defaultdict(list)
        self.vocabulary = set()       
        self.max = 1 

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        words = text.split()
        return words    

    def pre_tokenize(self, file_path):
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i >= 96255:
                    break
                article = json.loads(line.strip())  
                
                # Extract title and abstract
                title = article.get("title", "")
                abstract = article.get("abstract", "")

                for text in [title, abstract]:
                    if "|" in text:
                        continue
                    words = self.preprocess_text(text)
                    for word in words:
                        vocab_from_text = []
                        word = word.encode('ascii', errors='ignore').decode('ascii')                        
                        if len(word)==0: 
                            continue             
                        if word in self.word_freq:
                            ls = self.word_freq[word]
                            self.word_freq[word] = [ls[0]+1, 0]
                        else:
                            self.word_freq[word] = [1, 0]
                        # if word not in self.word_split:
                            vocab_from_text.append(word[0])
                            for letter in word[1:]:
                                new_letter = "|" + letter                    
                                vocab_from_text.append(new_letter)                    
                            self.word_split[word] = vocab_from_text
                            self.vocabulary.update(vocab_from_text)
        # vocab_list = sorted(vocabulary)

        # print(self.word_freq)
    
    def compute_pair_scores(self):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        flag = 0
        for word, value in self.word_freq.items():
            if value[1] < self.max:
                # continue
                flag = 1
                freq = value[0]
                split = self.word_split[word]
                if len(split) == 1:
                    letter_freqs[split[0]] += freq
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    letter_freqs[split[i]] += freq
                    pair_freqs[pair] += freq
                letter_freqs[split[-1]] += freq

        if len(pair_freqs)==0:
            self.max=self.max+1
            return self.compute_pair_scores()
        
        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        top_pair = max(scores, key=scores.get)
        return scores, top_pair
    
    def merge_pair(self, pair):
        for word in self.word_freq:
            split = self.word_split[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    merge = pair[0] + pair[1][1:] if pair[1][0]=="|" else pair[0] + pair[1]
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            self.word_split[word] = split
            ls = self.word_freq[word]
            upd_ls = [ls[0], ls[1]+1]
            self.word_freq[word] = upd_ls
    
    def make_all_merges(self, desired_vocab_size, verbose):
        num_merges = desired_vocab_size - len(self.vocabulary)
        i = 0
        while i < num_merges:
            start = time.time()
            scores, top_pair = self.compute_pair_scores()
            self.merge_pair(top_pair)
            new_token = (
                top_pair[0] + top_pair[1][1:]
                if top_pair[1][0]=="|"
                else top_pair[0] + top_pair[1]
            )
            # vocab_list.append(new_token)
            self.vocabulary.add(new_token)
            # print(time.time()-start)
            # if verbose==1 : 
            #     if i%100==0: print(f"{i+1} merges done")
            i += 1

    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocabulary:
                i -= 1
            if i == 0:
                return ["[OOD]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"|{word}"
        return tokens
    
    def write(self, vocab_file_path):
        with open(vocab_file_path, "w") as file:
            for token in self.vocabulary:
                file.write(token + "\n")

        # print(f"Tokens have been saved to {vocab_file_path}")
        
    def train_tokenizer(self, file_path, desired_vocab_size, verbose):
        self.pre_tokenize(file_path)
        self.make_all_merges(desired_vocab_size, verbose)
        # print(self.max)
        # print(self.word_freq)
        # print(self.vocabulary)
        # print(self.word_freq)
        # print(self.word_split)

    def get_tokens(self, text):
        upd_text = self.preprocess_text(text)
        upd_text = [word for word in upd_text]
        encoded_words = [self.encode_word(word) for word in upd_text]
        return sum(encoded_words, [])


class InvertedIndexBinary:
    def __init__(self, tokenizer_choice, verbose, desired_vocab_size):
        self.verbose = verbose
        self.vocabulary = {}
        self.postings = defaultdict(list)
        self.next_token_id = 1
        self.tokenizer_choice = tokenizer_choice
        self.doc_mapping = {}

        if self.tokenizer_choice==1:
            self.tokenizer = SimpleTokenizer()
            tokenizer_start = time.time()
            if self.verbose==1 : print(self.tokenizer)
            self.tokenizer.train_tokenizer(file_path=doc_file)
            if self.verbose==1 : print(f"Tokenizer training completed in {time.time()-tokenizer_start}")            
        elif self.tokenizer_choice==2:
            self.tokenizer = BPE()
            if self.verbose==1 : print(self.tokenizer)
            tokenizer_start = time.time()
            self.tokenizer.train_tokenizer(file_path=doc_file, desired_vocab_size=desired_vocab_size, verbose=self.verbose)
            if self.verbose==1 : print(f"Tokenizer training completed in {time.time()-tokenizer_start}")            
            self.tokenizer.write_merges("merges_BPE.txt")
        elif self.tokenizer_choice==3:
            self.tokenizer = WordPiece()
            if self.verbose==1 : print(self.tokenizer)
            tokenizer_start = time.time()
            self.tokenizer.train_tokenizer(file_path=doc_file, desired_vocab_size=desired_vocab_size, verbose=self.verbose)
            if self.verbose==1 : print(f"Tokenizer training completed in {time.time()-tokenizer_start}")
            self.tokenizer.write("vocabulary_WordPiece.txt")


    def tokenize(self, content):
        tokenized_text = self.tokenizer.get_tokens(content)
        return tokenized_text       

    def add_document(self, doc_id, content):
        # token_start = time.time()
        article = json.loads(line.strip())
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        token_freq = defaultdict(int)
        content = title + abstract
        tokens = self.tokenize(content)
        for token in tokens:
            token_freq[token] += 1
        for token, freq in token_freq.items():
            if token not in self.vocabulary:
                self.vocabulary[token] = self.next_token_id
                self.next_token_id += 1
            token_id = self.vocabulary[token]
            self.postings[token_id].append((doc_id, freq))
        if self.verbose==1:
            if doc_id%1000==0:
                print(f'Indexing done till doc_id : {doc_id}')
        self.doc_mapping[doc_id] = article.get("doc_id", "")

    def write_to_disk(self, vocab_file, index_file):
        with open(vocab_file, 'w') as vf:
            for token, token_id in sorted(self.vocabulary.items(), key=lambda x: x[1]):
                vf.write(f"{token_id}, {token}\n")
        
        with open(index_file, 'wb') as pf:
            for token_id, postings in sorted(self.postings.items()):
                pf.write(struct.pack('I', token_id))  
                pf.write(struct.pack('I', len(postings)))
                
                for doc_id, positions in postings:
                    pf.write(struct.pack('I', doc_id))  
                    pf.write(struct.pack('I', positions))

    def write_doc_id(self, doc_map_file):
        with open(doc_map_file, 'w') as file:
            for doc_num, doc_id in self.doc_mapping.items():
                file.write(f"{doc_num}:{doc_id}\n")
        
    def write_to_disk1(self, vocab_file, postings_file):
        with open(vocab_file, 'w') as vf:
            for token, token_id in sorted(self.vocabulary.items(), key=lambda x: x[1]):
                vf.write(f"{token_id}, {token}\n")
        
        with open(postings_file, 'w') as pf:
            for token_id, postings in sorted(self.postings.items()):
                postings_str = ' '.join(
                    f"{doc_id}: {positions}" for doc_id, positions in postings)
                pf.write(f"{token_id}, {postings_str}\n")

start_time = time.time()
verbose=0

doc_file = sys.argv[1]
index_file = sys.argv[2]
tokenizer_choice = sys.argv[3]
tokenizer_choice = int(tokenizer_choice) + 1

# tokenizer_choice = 3

tokenizer_choice_file = "tokenizer_choice_file.txt"
with open(tokenizer_choice_file, "w") as file:
    file.write(f"{tokenizer_choice}")

# doc_file = "train_data\\cord19-trec_covid-docs.txt"
# doc_file = "train_data\\rough.txt"

inverted_index = InvertedIndexBinary(tokenizer_choice, desired_vocab_size = 1600, verbose=verbose)

indexing_start = time.time()
with open(doc_file, "r") as file:
    for i, line in enumerate(file):
        inverted_index.add_document(i+1, line)

end_time = time.time()

if verbose==1 : print(f"Indexing completed in {end_time-indexing_start}")

if verbose==1 : print(f"Completed in {end_time-start_time}")

# index_file = "index"
vocab_file = f"{index_file}"+".dict"
postings_file = f"{index_file}"+".idx"
inverted_index.write_to_disk(vocab_file, postings_file)

inverted_index.write_doc_id("doc_map.txt")







