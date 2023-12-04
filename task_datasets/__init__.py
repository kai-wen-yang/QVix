DATA_DIR = '/fs/nexus-scratch/kwyang3/data'

from .vqa_datasets import ScienceQADataset
from functools import partial


dataset_class_dict = {
    # VQA Datasets
    'ScienceQA': ScienceQADataset,
}
