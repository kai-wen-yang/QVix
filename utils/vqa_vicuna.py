import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from .tools import VQAEval
import pdb
import sys
sys.path.append("..")
from models import get_image
from .prompt import prompt_template
import collections
from datasets import Dataset, load_dataset


vicuna_prompt =\
    'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\\nUSER: {}\nASSISTANT:'


def evaluate_VQA_vicuna(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers',
    max_new_tokens=256,
    args=None,
    classnames=None
):
    vqa_gpt_prompt = prompt_template[args.prompt]
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

    for t, batch in enumerate(tqdm(dataloader, desc="Running inference")):

        questions = []
        messages = []
        for i in range(len(batch['image_path'])):
            message = vicuna_prompt.format(vqa_gpt_prompt.format(f"{batch['question'][i]}\n"))
            messages.append(message)

        reply = model.text_generate(messages)
        subquestions = reply
        for i in range(len(batch['image_path'])):
            question = reply[i]+'\n'+args.question.format(batch['question'][i])
            questions.append(question)

        outputs = model.batch_generate(batch['image_path'], questions, max_new_tokens=max_new_tokens)
        for image_path, question, gt_answer, output, subject, grade, subque in zip(batch['image_path'], questions, batch['gt_answers'], outputs, batch['subject'], batch['grade'], subquestions):
            split_output = output.split(',')[0]
            answer_dict={'question': question, 'answer': split_output, 'ori_answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name, 'subject': subject, 'grade': grade}
            predictions.append(answer_dict)
   
        text_table = wandb.Table(columns=["answer", "label",  "question", "ori_answer"])
        if type(gt_answer) is not str:
            gt_answer = gt_answer[0]
        text_table.add_data(split_output, gt_answer, question, output)
        wandb.log({f'time{time}_batch{str(t)}/image.jpg': wandb.Image(get_image(image_path)),
                   f'time{time}_batch{str(t)}/table': text_table,
                   })

    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    eval = VQAEval()
    correct, correct_nat, correct_soc, correct_lan, correct_g16, correct_g712 = 0, 0, 0, 0, 0, 0
    num, num_nat, num_soc, num_lan, num_g16, num_g712 = 0, 0, 0, 0, 0, 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            if eval.evaluate(answer, gt_answers)==1:
                correct+=1
                if dict[i]['subject'] == 'natural science':
                    correct_nat+=1
                    num_nat+=1
                if dict[i]['subject'] == 'social science':
                    correct_soc+=1
                    num_soc+=1
                if dict[i]['subject'] == 'language science':
                    correct_lan+=1
                    num_lan+=1
                if dict[i]['grade'] in ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']:
                    correct_g16+=1
                    num_g16+=1
                if dict[i]['grade'] in ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']:
                    correct_g712+=1
                    num_g712+=1
            else:
                if dict[i]['subject'] == 'natural science':
                    num_nat+=1
                if dict[i]['subject'] == 'social science':
                    num_soc+=1
                if dict[i]['subject'] == 'language science':
                    num_lan+=1
                if dict[i]['grade'] in ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']:
                    num_g16+=1
                if dict[i]['grade'] in ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']:
                    num_g712+=1
            num+=1
    results = {
        'acc': float(correct)/num,
        'acc_nat': float(correct_nat)/num_nat,
        'acc_soc': float(correct_soc)/num_soc,
        'acc_lan': float(correct_lan)/num_lan,
        'acc_g16': float(correct_g16)/num_g16,
        'acc_g712': float(correct_g712)/num_g712,
    }
    print(results)
    wandb.log(results)

    return results