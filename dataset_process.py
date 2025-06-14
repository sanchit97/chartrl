import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.nn.parallel import DataParallel
import torch.nn.functional as F
import pickle
from tqdm import tqdm

from datasets import Dataset, load_dataset, load_from_disk
from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoModelForVisualQuestionAnswering,AutoModelForCausalLM, AutoModelForZeroShotObjectDetection
from transformers import pipeline

from qwen_vl_utils import process_vision_info

from models import load_vlm_model

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# Caching
cache_dir = '/mnt/data/sanchit/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/data/sanchit/hf'
os.environ['TRANSFORMERS_CACHE']= '/mnt/data/sanchit/hf'
os.environ['HF_HOME'] = '/mnt/data/sanchit/hf'


class ChartDataset:
    def __init__(self, dataset_name, processor=None):
        self.dataset_name = dataset_name
        # self.dataset = self.load_chart_dataset(dataset_name, split, bsz)
        self.processor = processor

        self.system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
        Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
        The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
        Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    def load_chart_dataset(self, split):
        if self.dataset_name == "chartqa":
            dataset = load_dataset("HuggingFaceM4/ChartQA", cache_dir = cache_dir)
            data = dataset[split]
            # test_dataloader = DataLoader(test_data, batch_size=1, collate_fn = collate_fn_chartqa)
            # return test_dataloader
            return data

    def collate_fn_chartqa(self, batch):
        # for quick evals
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

    def train_collate_fn_chartqa(self, examples): 
        processor = self.processor  # Use the processor from the class instance, this is messy though #TODO
        # For SFT/RL
        # Get the texts and images, and apply the chat template
        # st = time.time()
        texts = [processor.apply_chat_template(self.format_question(example), tokenize=False) for example in examples]  # Prepare texts for processing
        # Process the images to extract inputs
        # just collect them into a list
        image_inputs = [process_vision_info(self.format_question(example))[0] for example in examples]
        # image_inputs = [[example['image']] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if "qwen" in processor.__class__.__name__.lower():  # Check if the processor is Qwen2VLProcessor
            # image_tokens = [151652, 151653, 151655]  # Specific image token IDs for (from the tutorial)
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(tok) for tok in processor.tokenizer.additional_special_tokens]

        elif "intern" in processor.__class__.__name__.lower():
            image_tokens = [92542, 92543, 92544, 92545, 92546]
        else: #llava type models
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        # logging.info("Time to process batch: ",time.time()   - st)  # Log the time taken for processing
        return batch  # Return the prepared batch

    def format_question(self, example):
        if "llava" in self.processor.__class__.__name__.lower():
            return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example["image"],
                    },
                    {
                        "type": "text",
                        "text": example["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["label"][0]}],
            },]
        else:
            return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example["image"],
                    },
                    {
                        "type": "text",
                        "text": example["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["label"][0]}],
            },]

    def load_pref_data(self):
        path = "./pref-data/base-llava-1.6-sft-hf-dset"
        if os.path.exists(path):
            pref_dataset = load_from_disk(path)
    
        else:
            logging.info("Pref Dataset not found, loading from pickle file... (can take a while)")
            with open("./pref-data/base-llava-1.6-sft", "rb") as f:
                pref_data = pickle.load(f)

            formatted_data = []
            for row in tqdm(pref_data):
                img = row[0]['image']
                query = row[0]['query']
                label = row[0]['label'][0]
                human = row[0]['human_or_machine']
                pred = row[1]
                formatted_data.append({
                    # "image": Image.open(img),
                    "image": img,
                    "question": query,
                    "label": label,
                    "human_or_machine": human,
                    "predicted": pred
                })
            # Create Hugging Face Dataset
            logging.info("Pref Data Processed")
            dataset = Dataset.from_list(formatted_data)
            pref_dataset = dataset.map(self.pref_format, num_proc=32)
            pref_dataset.save_to_disk("./pref-data/base-llava-1.6-sft-hf-dset")
            logging.info("Pref Dataset Created")
        
        return pref_dataset

    def pref_format(self, example):
        # Prepare the input for the chat template
        prompt = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": example["question"]}],
            },
        ]
        chosen = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["label"]}],
            },
        ]
        rejected = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["predicted"]}],
            },
        ]
        # Apply the chat template
        prompt = self.processor.apply_chat_template(prompt, tokenize=False)
        chosen = self.processor.apply_chat_template(chosen, tokenize=False)
        rejected = self.processor.apply_chat_template(rejected, tokenize=False)
        # Resize the image to ensure it fits within the maximum allowable
        # size of the processor to prevent OOM errors.
        # max_size = self.processor.image_processor.size["longest_edge"]
        # example["image"].thumbnail((max_size, max_size))
        return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}