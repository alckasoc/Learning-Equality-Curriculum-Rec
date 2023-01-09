#!/bin/bash
# for i in 0 1 2 3 4
# do
#    python baseline.py --epochs 5 --model "sentence-transformers/all-MiniLM-L6-v2" --project_run_root "all-MiniLM-L6-v2" --fold $i --patience 1 --debug 0
# done

for i in 0 1 2 3 4
do
   python baseline.py --epochs 10 --model "xlm-roberta-base" --project_run_root "xlm-roberta-base" --fold $i --patience -1 --debug 0
done