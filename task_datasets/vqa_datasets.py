import os
import json
import datasets
from torch.utils.data import Dataset
from . import DATA_DIR
import pdb
from datasets import load_dataset, Image, load_from_disk
    

class ScienceQADataset(Dataset):
    split='test'
    options = ["A", "B", "C", "D", "E", "F", "G", "H"]
    data_root = f'{DATA_DIR}/VQA_Datasets/ScienceQA'

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        self.subject_list = []
        self.grade_list = []
        ann_path = f"{self.data_root}/{self.split}_anns.json"


        if os.path.exists(ann_path):
            dataset = json.load(open(ann_path, "r"))
            for sample in dataset:
                self.image_list.append(sample['image_path'])
                self.question_list.append(sample['question'])
                self.answer_list.append(sample['answer'])
                self.subject_list.append(sample['subject'])
                self.grade_list.append(sample['grade'])
        else:
            self.load_save_dataset()
    
    def load_save_dataset(self):
        # load dataset
        data = datasets.load_dataset('derek-thomas/ScienceQA', self.split).cast_column("image", Image(decode=True))
        for sample in data[self.split]:
            if sample['image'] is None:
                continue
            # question = f"Question: {sample['question']}\n" \
            #            f"Options: {' '.join([f'({x}) {y}' for x, y in zip(self.options, sample['choices'])])}\n"
            options = "\n".join(sample['choices'])
            question = f"{sample['question']}\n" \
                       f"Options: {options}\n"

            self.question_list.append(question)
            self.image_list.append(sample['image'].convert('RGB'))
            self.answer_list.append(sample['choices'][sample['answer']])
            self.subject_list.append(sample['subject'])
            self.grade_list.append(sample['grade'])

        # save dataset
        dataset = []
        for i in range(len(self.image_list)):
            img_file_name = f'{self.data_root}/{self.split}_imgs/{i:04d}.png'
            if not os.path.exists(img_file_name):
                self.image_list[i].save(img_file_name)
            self.image_list[i] = img_file_name
            dataset.append({
                'answer': self.answer_list[i],
                'image_path': self.image_list[i],
                'question': self.question_list[i],
                'subject': self.subject_list[i],
                'grade': self.grade_list[i]
            })
        with open(f"{self.data_root}/{self.split}_anns.json", "w") as f:
            f.write(json.dumps(dataset, indent=4))

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx: int):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers,
            "subject": self.subject_list[idx],
            "grade": self.grade_list[idx]}

