#!/bin/bash
if ! command -v python3 &>/dev/null; then
    echo "Python3 could not be found. Please install Python3."
    exit 1
fi

chmod +x w2v_local_rerank.py
chmod +x w2v_gen_rerank.py
chmod +x glove_gen_rerank.py
chmod +x w2v-local_rerank.sh
chmod +x w2v-gen_rerank.sh
chmod +x glove-gen_rerank.sh

echo "Build completed."