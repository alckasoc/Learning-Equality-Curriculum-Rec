#!/bin/bash
for i in 1 2 3 4 5
do
   python train.py --fold $i --patience 1
done

