import os
import pathlib
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.nn.parallel import DataParallel
import torch.nn.functional as F

os.environ["FLASH_ATTENTION_2_ENABLED"] = "1"

from datasets import Dataset, load_dataset, load_from_disk

from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoModelForVisualQuestionAnswering,AutoModelForCausalLM, AutoModelForZeroShotObjectDetection
from transformers import pipeline

from transformers import DonutProcessor, VisionEncoderDecoderModel

from transformers import pipeline

import argparse
from tqdm import tqdm
import os
import json
import numpy as np
# from openai import OpenAI
import warnings
import pickle
from undecorated import undecorated
from types import MethodType
from PIL import Image




# Caching
cache_dir = '/mnt/data/sanchit/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/data/sanchit/hf'
os.environ['TRANSFORMERS_CACHE']= '/mnt/data/sanchit/hf'
os.environ['HF_HOME'] = '/mnt/data/sanchit/hf'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vlm_model(model_type):
    # For smaller models (CLIP based)
    if model_type== "blip2":
        model_name = "Salesforce/blip2-flan-t5-xl"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name,device_map = "auto", cache_dir = cache_dir)
    if model_type == "instructblip":
        processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", device_map="auto", cache_dir = cache_dir)
    if model_type == "blip2-xxl":
        processor = InstructBlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="auto",cache_dir = cache_dir)
    
    if model_type == "pali-3b":
        model_name = "google/paligemma2-3b-mix-448"
        processor = AutoProcessor.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)


    # For medium models (below 8b)
    if model_type == "instructblip-vicuna":
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map="auto",cache_dir = cache_dir)
    if model_type == "llava-1.5":
        model_name = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
    if model_type == "llava-1.6":
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir, attn_implementation = "flash_attention_2", torch_dtype=torch.bfloat16)
    if model_type == "qwen-7b":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto",cache_dir = cache_dir, attn_implementation = "flash_attention_2", torch_dtype=torch.bfloat16)
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
        # model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-7B", device_map="auto",cache_dir = cache_dir, attn_implementation = "flash_attention_2", torch_dtype=torch.bfloat16)
    if model_type == "internvl-8b":
        processor = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-8B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True,torch_dtype=torch.bfloat16)
        model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-8B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True,torch_dtype=torch.bfloat16)

    # For big models (above 13-15b)
    if model_type == "instructblip-xxl": #prob 13b
        processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
        model = AutoModelForImageTextToText.from_pretrained("Salesforce/instructblip-flan-t5-xxl", device_map="auto",cache_dir = cache_dir)
    if model_type == "instructblip-vicuna-13b":
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map="auto",cache_dir = cache_dir)
    if model_type == "llava-1.5-13b":
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
        model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-1.5-13b-hf", device_map="auto",cache_dir = cache_dir)
    if model_type == "llava-1.6-13b":
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
        model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", device_map="auto",cache_dir = cache_dir)


    # For SOTA models (very big)
    if model_type == "internvl-26b":
        processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2_5-26B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-26B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True)
    


    # Chart specific models
    if model_type == "unichart-chartqa":
        model_name = "ahmed-masry/unichart-chartqa-960"
        model = VisionEncoderDecoderModel.from_pretrained(model_name).cuda()
        processor = DonutProcessor.from_pretrained(model_name)
        # default_cfg = model.config_class()
        # for k, v in default_cfg.to_dict().items():
        #     model.config.__dict__.setdefault(k, v)
        out_dir = pathlib.Path(cache_dir) / model_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        model.save_pretrained(out_dir, safe_serialization=True)   # .safetensors + new config
        processor.save_pretrained(out_dir)
        model = VisionEncoderDecoderModel.from_pretrained(out_dir, torch_dtype="auto")
        processor = AutoTokenizer.from_pretrained(out_dir)


    if model_type == "chart-gemma":
        model_type = "ahmed-masry/chartgemma"
        model = AutoModelForImageTextToText.from_pretrained(model_type, device_map="auto",cache_dir = cache_dir)
        processor = AutoProcessor.from_pretrained(model_type)
    


    # OCR models (pipelines)
    if model_type == "generic-ocr":
        pipe = pipeline("image-text-to-text", model="Qwen/Qwen2-VL-2B-Instruct")
        return pipe

    return model, processor




    
    