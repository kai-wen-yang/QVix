#!/bin/bash

cd /QVix/models
export PYTHONPATH="$PYTHONPATH:$PWD"
cd ..

python tools/eval.py --model_name InstructBLIP7B --batch_size 4 --dataset_name ScienceQA --device 0 --expname 'baseline' --sample_num 1000 --task_name vqa --question "Question: {}Answer:" 

