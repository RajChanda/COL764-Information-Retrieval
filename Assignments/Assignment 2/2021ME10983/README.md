Build Script (*build.sh*)
```
bash build.sh
```

For running the task0 code and generate output in TREC Format use this:-
```
bash task0.sh <query_file_path> <query_top_100> <docs_path> <output_file_path>
```

For running the task1 code - Locally Trained Word2Vec Embeddings:-
```
bash w2v-local_rerank.py <query_file_path> <query_top_100> <docs_path> <output_file_path> <expansions_file_path>
```


For running the task2 code - Pre-Trained Word2Vec Embeddings:-
```
bash w2v-gen_rerank.py <query_file_path> <query_top_100> <docs_path> <w2v-embeddings_file_path> <output_file_path> <expansions_file_path>
```

For running the task2 code - Pre-Trained Glove Embeddings:-
```
bash w2v-gen_rerank.py <query_file_path> <query_top_100> <docs_path> <glove-embeddings_file_path> <output_file_path> <expansions_file_path>
```

*output_file_path* will contain the output in the TREC format
*expansions_file_path* will contain the expansion terms for the queries

