All desired path files have to be put in by the user

$ unzip 2021ME10983.zip
$ cd 2021ME10983
$ bash build.sh
$ bash dictcons.sh [coll-path] {0|1|2}
$ bash invidx.sh [coll-path] [index-file] {0|1|2}
$ bash tf_idf_search.sh [queryfile] [resultfile] [indexfile] [dictfile]