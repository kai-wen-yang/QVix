import os
import json
import argparse
import datetime
from functools import partial

import torch
import numpy as np
import sys
sys.path.append("..")
from utils import task_class_dict
from task_datasets import dataset_class_dict
from models import get_model
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="InstructBLIP7B")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=-1)

    # datasets
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_seed", type=int, default=0)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    # QVix
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default='prompt_hand_v1')
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        try:
            classnames = dataset.classnames
        except:
            classnames = None
        return dataset, classnames

    if len(dataset) > max_sample_num:

        drop_size = len(dataset) - max_sample_num
        dataset, _ = torch.utils.data.random_split(dataset, [max_sample_num, drop_size],
                                                                    generator=torch.Generator().manual_seed(42))
        try:
            classnames = dataset.dataset.classnames
        except:
            classnames = None
    return dataset, classnames


def get_eval_function(args):
    eval_func = task_class_dict[args.task_name]

    if args.max_new_tokens != -1:
        eval_func = partial(eval_func, max_new_tokens=args.max_new_tokens)

    return eval_func


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, args)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    eval_function = get_eval_function(args)
    answer_path = f"{args.answer_path}/{args.model_name}/{args.dataset_name}/{args.task_name}/{args.expname}"
    args.expname = args.dataset_name + '_'+ args.model_name +'_'+ args.task_name + '_'+ args.expname
    wandb.init(name=args.expname, config=args)

    if eval_function is not None:
        dataset = dataset_class_dict[args.dataset_name]()
        dataset, classnames = sample_dataset(dataset, args.sample_num, args.sample_seed)
        results = eval_function(model, dataset, args.model_name, args.dataset_name, time, args.batch_size, answer_path=answer_path, args=args, classnames=classnames)
    
    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)
