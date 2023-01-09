#!/bin/bash
for i in 0
do
   python baseline.py --model "xlm-roberta-base" --project_run_root "xlm-roberta-base" --fold $i --patience 1 --debug 1
done

