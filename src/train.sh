#!/bin/bash
# for i in 0 1 2 3 4
# do
#    python baseline.py --epochs 5 --model "sentence-transformers/all-MiniLM-L6-v2" --project_run_root "all-MiniLM-L6-v2" --fold $i --patience 1 --debug 0
# done

# for i in 0 1 2 3 4
# do
#    python baseline.py --epochs 10 --model "xlm-roberta-base" --project_run_root "xlm-roberta-base" --fold $i --patience -1 --debug 0
# done

# python baseline.py 
#     --epochs 5 --model "sentence-transformers/all-MiniLM-L6-v2" \
#     --correlations "../input/correlations.csv" --train "../input/train_5fold.csv" \
#     --project "LECR_0.297_baseline" --project_run_root "all-MiniLM-L6-v2" \
#     --save_root "../models/0297_baseline/" \
#     --fold 0 --patience 1 --debug 0 --gradient_checkpointing 1

# for i in 0 1 2 3 4
# do
#    python baseline.py --epochs 5 --model "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" --project_run_root "paraphrase-multilingual-mpnet-base-v2" --fold $i --patience -1 --debug 1 --gradient_checkpointing 1
# done

# for i in 0
# do
#    python baseline.py --epochs 10 --model "xlm-roberta-base" --project_run_root "xlm-roberta-base_ep10" --fold $i --patience -1 --debug 0 --gradient_checkpointing 1
# done

for i in 0 1 2 3 4
do
   python baseline.py --epochs 5 --model "sentence-transformers/all-MiniLM-L6-v2" --correlations "../input/correlations.csv" --train "../input/train_better_cv_5fold.csv" --project "LECR_better_cv_test" --project_run_root "all-MiniLM-L6-v2" --save_root "../models/better_cv_test/" --fold $i --patience 1 --debug 0 --gradient_checkpointing 1
done