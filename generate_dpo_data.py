import os
import torch
import pickle
import json
from tqdm import tqdm

from models import load_vlm_model
from dataset_process import ChartDataset

import random

from metrics import exact_match, relaxed_accuracy
from utils import get_vlm_output, clear_memory, format_data


system_message = """You are an OCR engine.

Extract every visible character **inside the plotted area** exactly as it appears.

1. Preserve original line breaks and the reading order (top-to-bottom, left-to-right).  
2. **Ignore** chart titles, axis names, legends, gridlines, and any decorative text.  
3. Focus only on numbers, data labels, and in-chart annotations.  
4. Return **only** the raw text – no summaries, no markdown, no extra tokens."""

def run_vlm_ocr(ocr_model, image):
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((200, int(image.height * (200 / image.width))))},
            ]
        },
    ]
    out = ocr_model(text=messages, max_new_tokens=200)
    # breakpoint()
    # out = out[]
    out = out[0]['generated_text'][2]['content'].strip().split()
    return out


def main():
    with open("./pref-data/base-llava-1.6-sft", "rb") as f:
        pref_data = pickle.load(f)

    ocr_pipe = load_vlm_model("generic-ocr")

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
        all_entity = run_vlm_ocr(ocr_pipe, row[0]['image'])
        for entity in all_entity:
            if entity.lower() != row[0]['label'][0].lower():
                img = row[0]['image']
                query = row[0]['query']
                label = row[0]['label'][0]
                human = row[0]['human_or_machine']
                pred = entity
                formatted_data.append({
                    # "image": Image.open(img),
                    "image": img,
                    "question": query,
                    "label": label,
                    "human_or_machine": human,
                    "predicted": pred
            })
    # Create Hugging Face Dataset
    # logging.info("Pref Data Processed")
    breakpoint()
    dataset = Dataset.from_list(formatted_data)

    dataset = pickle.load(open("./pref-data/base-llava-1.6-sft-dataset-hardneg.pkl", "rb"))
    pref_dataset = dataset.map(pref_format, num_proc=32)
    pref_dataset.save_to_disk("./pref-data/base-llava-1.6-sft-hf-dset-hard-negatives")


from transformers import LlavaForConditionalGeneration, LlavaNextProcessor
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_name)


def pref_format(example):
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
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
    # Resize the image to ensure it fits within the maximum allowable
    # size of the processor to prevent OOM errors.
    # max_size = self.processor.image_processor.size["longest_edge"]
    # example["image"].thumbnail((max_size, max_size))
    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

main()