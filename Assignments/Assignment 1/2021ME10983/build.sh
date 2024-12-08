#!/bin/bash
if ! command -v python3 &>/dev/null; then
    echo "Python3 could not be found. Please install Python3."
    exit 1
fi

chmod +x dict_cons.py
chmod +x invidx_cons.py
chmod +x tf_idf_search.py
chmod +x dictcons.sh
chmod +x invidx.sh
chmod +x tf_idf_search.sh

echo "Build completed."
