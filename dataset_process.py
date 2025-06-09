import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.nn.parallel import DataParallel
import torch.nn.functional as F

from datasets import Dataset, load_dataset, load_from_disk

# Caching
cache_dir = '/mnt/data/sanchit/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/data/sanchit/hf'
os.environ['TRANSFORMERS_CACHE']= '/mnt/data/sanchit/hf'
os.environ['HF_HOME'] = '/mnt/data/sanchit/hf'

def collate_fn_chartqa(batch):
    # breakpoint()
    image_batch = []
    query_batch = []
    label_batch = []
    machine_human_batch = []
    
    for idx in range(len(batch)):
        image_batch.append(batch[idx]['image'])
        query_batch.append(batch[idx]['query'])
        label_batch.append(batch[idx]['label'])
        machine_human_batch.append(batch[idx]['human_or_machine'])


    return image_batch, query_batch, label_batch, machine_human_batch

def load_chart_dataset(dataset_name, split = "test", bsz=1):
    if dataset_name == "chartqa":
        dataset = load_dataset("HuggingFaceM4/ChartQA", cache_dir = cache_dir)
        test_data = dataset[split]
        test_dataloader = DataLoader(test_data, batch_size=1, collate_fn = collate_fn_chartqa)

        return test_dataloader