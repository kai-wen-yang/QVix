from .vqa import evaluate_VQA
from .vqa_gpt import evaluate_VQA_gpt
from .vqa_vicuna import evaluate_VQA_vicuna

task_class_dict = {
'vqa': evaluate_VQA,
'vqa_gpt': evaluate_VQA_gpt,
'vqa_vicuna': evaluate_VQA_vicuna,
}