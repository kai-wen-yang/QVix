#!/bin/bash

cd /QVix/models
export PYTHONPATH="$PYTHONPATH:$PWD"
cd ..

python tools/eval.py --model_name InstructBLIP7B --batch_size 4 --dataset_name ScienceQA --device 0 --expname 'QVix' --sample_num 1000 --task_name vqa_gpt --prompt prompt_hand_v1 --cot "Question: {}Answer:" --api_key ' '