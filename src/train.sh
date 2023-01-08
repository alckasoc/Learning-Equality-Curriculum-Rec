#!/bin/bash
for i in 0 1 2 3 4
do
   python 0297_baseline.py --fold $i --patience 1
done

