#!/bin/bash
for i in 0
do
   python 0297_baseline.py --model "xlm-roberta-large" --project_run_root "xlm-roberta-large" --fold $i --patience 1
done

