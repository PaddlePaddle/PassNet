#!/bin/bash
set -x

# cd /ai4c/root/directory
if [ $(cat .git/config | grep 'ai4c' | wc -l) -eq 0 ]; then
    echo Please change to ai4c repo root directory
    exit 1
fi


get_selected() {
    head -200 ./graph_lists/fusible_subgraphs.txt
}

get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}' \
    | xargs -I{} mkdir -p {}

get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8"/pass_dir"}' \
    | xargs -I{} mkdir -p {}

get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8"/pass_dir"}' \
    | xargs -I{} touch ./{}/.ignore

get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}' \
    | xargs -I{} unlink ./{}/graphs
get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}' \
    | xargs -I{} ln -sr ./graphs ./{}/graphs

get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}' \
    | xargs -I{} unlink ./{}/entry.sh
get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}' \
    | xargs -I{} ln -sr ./entry_scripts/entry.sh ./{}/entry.sh

get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}' \
    | xargs -I{} unlink ./{}/graph_net_bench
get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}' \
    | xargs -I{} ln -sr ./graph_net_bench ./{}/graph_net_bench

get_selected | while read i;
do
    filepath=$(echo $i | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}')/graph_list.txt
    echo graphs/$i > $filepath
done

get_selected \
    | awk -F'/' '{print "samples/fusible_subgraphs/"$4"/"$8}' \
    | tee ./sample_lists/demo_fusible_samples.txt

