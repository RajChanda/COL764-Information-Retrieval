import re
import json
from collections import defaultdict
import time
import sys

class SimpleTokenizer:
    def __init__(self):
        self.vocabulary = defaultdict(int)

    def tokenizer(self, text):
        pattern = r'[ ,.:;"\']'
        tokens = re.split(pattern, text)
        tokens = [token.lower() for token in tokens if token]
        for token in tokens:
            self.vocabulary[token] += 1

    def process_articles(self, file_path, fields=["title", "abstract"]):
        with open(file_path, 'r') as file:
            for line in file:
                article = json.loads(line.strip())
                
                for field in fields:
                    field_text = article.get(field, "")  
                    field_text = field_text.encode('ascii', errors='ignore').decode('ascii')
                    self.tokenizer(field_text)
    
    def get_tokens(self, text):
        pattern = r'[ ,.:;"\']'
        tokens = re.split(pattern, text)
        for i in range(len(tokens)):
            tokens[i] = tokens[i].encode('ascii', errors='ignore').decode('ascii').lower()
        return tokens
    
    def train_tokenizer(self, file_path):
        self.process_articles(file_path)

    def write(self, vocab_file_path):
        with open(vocab_file_path, "w") as file:
            for token in self.vocabulary:
                # print(token)
                file.write(token + "\n")

        if verbose==1: print(f"Tokens have been saved to {vocab_file_path}")


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
        initial_corpus = defaultdict(int)
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
                        word = word.encode('ascii', errors='ignore').decode('ascii').encode('utf-8')
                        letter_list = list(map(int, word))
                        if element in self.corpus:
                            ls = self.corpus[element]
                            ls[-1] += 1
                            self.corpus[element] = ls
                        else:
                            ls = letter_list
                            ls.append(1)
                            self.corpus[element] = ls
                        initial_vocab.update(word)
        self.vocabulary = sorted(initial_vocab)

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
        self.vocabulary.append(idx)
    
    def make_all_merges(self, desired_vocab_size, verbose):
        num_merges = desired_vocab_size - len(self.vocabulary)
        for i in range(num_merges):
            # start = time.time()
            counts, top_pair = self.get_pair_freq()
            idx = 256 + i
            if verbose==1 : 
                if i%100==0: print(f"{i+1} merges done")
            self.merge_op(top_pair, idx)
            self.encoded_merges[top_pair] = idx

    def decode(self, ids):
        tokens = b"".join(self.vocab1[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        words = self.preprocess_text(text)
        encoded_text = []
        for word in words:
            word = word.encode('ascii', errors='ignore').decode('ascii').encode('utf-8')
            tokens = list(map(int, word))
            while len(tokens) >= 2:
                counts = {}
                for pair in zip(tokens, tokens[1:]):
                    counts[pair] = counts.get(pair, 0)+1
                pair = min(counts, key=lambda p: self.encoded_merges.get(p, float("inf")))
                if pair not in self.encoded_merges:
                    break 
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

        if verbose==1: print(f"Tokens have been saved to {vocab_file_path}")

    def write_merges(self, merge_file_path):
        with open(merge_file_path, "w") as file:
                for key, value in self.merges.items():
                    file.write(f"{key[0]},{key[1]}:{value}\n")
        if verbose==1: print(f"Merges have been saved to {merge_file_path}")

    def get_tokens(self, text):
        tokenized_text = []
        upd_text = self.preprocess_text(text)
        upd_text = [word for word in upd_text]
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
        # while time.time()-start_time < 280:
        while i < num_merges:
            # start = time.time()
            scores, top_pair = self.compute_pair_scores()
            self.merge_pair(top_pair)
            new_token = (
                top_pair[0] + top_pair[1][1:]
                if top_pair[1][0]=="|"
                else top_pair[0] + top_pair[1]
            )
            self.vocabulary.add(new_token)
            i += 1
            # print(f"{i} merges are done")

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

        if verbose==1 : print(f"Tokens have been saved to {vocab_file_path}")
        
    def train_tokenizer(self, file_path, desired_vocab_size, verbose):
        self.pre_tokenize(file_path)
        self.make_all_merges(desired_vocab_size, verbose)

    def get_tokens(self, text):
        upd_text = self.preprocess_text(text)
        upd_text = [word for word in upd_text]
        encoded_words = [self.encode_word(word) for word in upd_text]
        return sum(encoded_words, [])

verbose = 1

file_path = sys.argv[1]
tokenizer_choice = sys.argv[2]
tokenizer_choice = int(tokenizer_choice) + 1
output_file_path = "output.dict"

# file_path = "train_data\\rough.txt"  
# file_path = "train_data\\cord19-trec_covid-docs.txt"
# tokenizer_choice = 3

start_time = time.time()
if tokenizer_choice==1:
    simple_tokenizer = SimpleTokenizer()
    simple_tokenizer.train_tokenizer(file_path)
    simple_tokenizer.write(output_file_path)
elif tokenizer_choice==2:
    bpe = BPE()
    bpe.train_tokenizer(file_path, desired_vocab_size=900, verbose = verbose)
    bpe.write_vocabulary(output_file_path)
    # bpe.write_merges("merges_BPE.txt")
elif tokenizer_choice==3:
    wordpiece = WordPiece()
    wordpiece.train_tokenizer(file_path, desired_vocab_size=400, verbose = verbose)
    wordpiece.write(output_file_path)
end_time = time.time()

# print(simple_tokenizer.vocabulary.keys())

if verbose==1 : print(f"time required for tokenizer = {end_time-start_time}")
