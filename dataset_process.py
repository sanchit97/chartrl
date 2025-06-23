import os
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.nn.parallel import DataParallel
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from pathlib import Path
import json

from datasets import Dataset, load_dataset, load_from_disk, Image, DatasetDict
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

# ChartQA
class ChartDataset:
    def __init__(self, dataset_name, processor=None):
        self.dataset_name = dataset_name
        # self.dataset = self.load_chart_dataset(dataset_name, split, bsz)
        self.processor = processor

        # self.system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
        # Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
        # The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
        # Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

        self.system_message = ""

    def load_chart_dataset(self, split):
        if self.dataset_name == "chartqa":
            dataset = load_dataset("HuggingFaceM4/ChartQA", cache_dir = cache_dir)
            data = dataset[split]
            # test_dataloader = DataLoader(test_data, batch_size=1, collate_fn = collate_fn_chartqa)
            # return test_dataloader
            return data

    def create_loader(self, data, bsz=1):
        dataloader = DataLoader(data, batch_size = bsz, collate_fn = self.collate_fn_chartqa)
        return dataloader

    def collate_fn_chartqa(self, batch):
        # for quick evals
        image_batch = []
        query_batch = []
        label_batch = []
        machine_human_batch = []
        
        for idx in range(len(batch)):
            image_batch.append(batch[idx]['image'])
            query_batch.append(batch[idx]['query'])
            label_batch.append(batch[idx]['label'][0])
            machine_human_batch.append(batch[idx]['human_or_machine'])

        return image_batch, query_batch, label_batch, machine_human_batch

    def train_collate_fn_chartqa(self, examples): 
        processor = self.processor  # Use the processor from the class instance, this is messy though #TODO
        # For SFT/RL
        # Get the texts and images, and apply the chat template
        # st = time.time()
        image_inputs = [process_vision_info(self.format_question(example))[0] for example in examples]

        for example in examples:
            example.pop('image')    # Remove the image key to avoid issues with the processor

        texts = [processor.apply_chat_template(self.format_question_text_only(example), tokenize=False, add_generation_prompt=True) for example in examples]  # Prepare texts for processing
        # Process the images to extract inputs
        # just collect them into a list
        # image_inputs = [[example['image']] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        batch["input_ids"] = batch["input_ids"]#.to(dtype=torch.long) 
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if "qwen" in processor.__class__.__name__.lower():  # Check if the processor is Qwen2VLProcessor
            # image_tokens = [151643, 151652, 151653, 151654, 151655, 151656]  # Specific image token IDs 
            image_tokens = [151643, 151652, 151653, 151654, 151655]
            # image_tokens = [processor.tokenizer.convert_tokens_to_ids(tok) for tok in processor.tokenizer.additional_special_tokens]
        elif "intern" in processor.__class__.__name__.lower():
            image_tokens = [92542, 92543, 92544, 92545, 92546]
        else: #llava type models
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch
        # logging.info("Time to process batch: ",time.time()   - st)  # Log the time taken for processing
        breakpoint()
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
    

    def format_question_text_only(self, example):
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
                        # "image": example["image"],
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
        # path = "./pref-data/base-llava-1.6-sft-hf-dset"
        path = "./pref-data/base-llava-1.6-sft-hf-dset-hard-negatives"
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


# PlotQA/FigQA
class PlotQADataset:
    def __init__(self, dataset_name, processor=None):
        self.dataset_name = dataset_name
        self.processor = processor
        self.system_message = ""


    def load_split(self, meta_file, img_dir):
        meta_file, img_dir = Path(meta_file), Path(img_dir)
        records: list[dict] = []
        with meta_file.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)                      # JSON array
            except json.JSONDecodeError:
                data = [json.loads(line) for line in f]  # JSON-Lines
        records = data  

        for r in tqdm(records['qa_pairs']):
            r["image"] = str(img_dir / f"{r['image_index']}.png")
            r["question_string"] = str(r["question_string"])
            r["answer"] = str(r["answer"])
            try:
                r.pop('template')
                r.pop('answer_bbox')
            except:
                pass

        return records['qa_pairs']

    def load_plotqa_dataset(self, split):
        if os.path.exists(cache_dir+"/plotqa_dataset"):
            chartqa = load_from_disk(str(cache_dir+"/plotqa_dataset"))

        else:
            plotqa_src = "./plotqa/"
            # ds = Dataset.from_list(examples)
            # ds = ds.cast_column("image", Image())

            print("Processing PlotQA dataset... (train)")
            train_ds = self.to_hf_dataset(self.load_split(plotqa_src+"plotqa-qa-train.json", plotqa_src+"img_train"))
            print("Processing PlotQA dataset... (val)")
            val_ds   = self.to_hf_dataset(self.load_split(plotqa_src+"plotqa-qa-val.json",   plotqa_src+"img_val"))
            print("Processing PlotQA dataset... (test)")
            test_ds  = self.to_hf_dataset(self.load_split(plotqa_src+"plotqa-qa-test.json",  plotqa_src+"img_test"))

            chartqa = DatasetDict({
                "train":      train_ds,
                "validation": val_ds,
                "test":       test_ds,
            })

            chartqa.save_to_disk(str(cache_dir+"/plotqa_dataset"))

        return chartqa[split]

    def to_hf_dataset(self, examples: list[dict]) -> Dataset:
        """
        Wrap the list of dicts in a Hugging Face Dataset and
        treat the 'image' column as a lazily-loaded Image feature.
        """        
        dset = Dataset.from_list(examples)
        dset = dset.cast_column("image", Image())
        return dset

    def collate_fn_plotqa(self, batch):
        # for quick evals
        image_batch = []
        query_batch = []
        label_batch = []
        type_batch = []
        
        for idx in range(len(batch)):
            image_batch.append(batch[idx]['image'])
            query_batch.append(batch[idx]['question_string'])
            label_batch.append(batch[idx]['answer'])
            type_batch.append(batch[idx]['type'])

        return image_batch, query_batch, label_batch, type_batch

    def create_loader(self, data, bsz=1):
        dataloader = DataLoader(data, batch_size = bsz, collate_fn = self.collate_fn_plotqa)
        return dataloader

    def train_collate_fn_plotqa(self, examples): 
        processor = self.processor  # Use the processor from the class instance, this is messy though #TODO
        # For SFT/RL
        # Get the texts and images, and apply the chat template
        # st = time.time()
        texts = [processor.apply_chat_template(self.format_question(example), tokenize=False) for example in examples]  # Prepare texts for processing
        # Process the images to extract inputs
        # just collect them into a list
        for example in examples:
            example['image'] = example['image'].resize((example['image'].width // 2, example['image'].height // 2)) # Resize images to reduce memory usage
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
        elif "llava" in processor.__class__.__name__.lower(): #llava type models
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
                        "text": example["question_string"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["answer"]}],
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
                        "text": example["question_string"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["answer"]}],
            },]


# Chart2Text / OpenCQA
class ChartToTextDataset:
    def __init__(self,
                dataset_root: str = "./chart-to-text/Chart-to-text-main/",
                label: str = "",
                seed: int = 2025,
                processor=None,
                cache_dir: str = cache_dir,
                val_ratio: float = 0.10,
                test_ratio: float = 0.20):

        self.dataset_root = Path(dataset_root)
        self.processor    = processor
        self.cache_dir    = Path(cache_dir)
        self.val_ratio    = val_ratio
        self.test_ratio   = test_ratio
        self.seed         = seed

    def load(self, split: str = "train") -> Dataset | DatasetDict:
        """
        Args
        ----
        split: "train" | "validation" | "test" | "all"
        Returns
        -------
        A Hugging Face Dataset OR DatasetDict (if split == "all").
        """
        cache_path = self.cache_dir / f"charttotext"
        if cache_path.exists():
            dset_dict = load_from_disk(str(cache_path))
        else:
            dset_dict = self._build_and_cache(cache_path)

        return dset_dict if split == "all" else dset_dict[split]

    def _build_and_cache(self, cache_path: Path) -> DatasetDict:
        examples_pew = self._gather_examples("pew_dataset/dataset/")
        examples_statista = self._gather_examples("statista_dataset/dataset/")
        examples = examples_pew + examples_statista
        full_ds = self._to_hf_dataset(examples)

        full_ds = full_ds.shuffle(seed=self.seed)
        n_total = len(full_ds)
        n_val   = int(n_total * self.val_ratio)
        n_test  = int(n_total * self.test_ratio)
        n_train = n_total - n_val - n_test

        train_ds      = full_ds.select(range(n_train))
        val_ds        = full_ds.select(range(n_train, n_train + n_val))
        test_ds       = full_ds.select(range(n_train + n_val, n_total))

        dset_dict = DatasetDict({
            "train":      train_ds,
            "validation": val_ds,
            "test":       test_ds,
        })

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dset_dict.save_to_disk(str(cache_path))
        return dset_dict

    def _gather_examples(self, name) -> list[dict]:
        """Collect (image, query, answer, label) rows from disk."""
        img_dir      = self.dataset_root / name / "imgs"
        caption_dir  = self.dataset_root / name / "captions"

        if not img_dir.is_dir() or not caption_dir.is_dir():
            raise FileNotFoundError(
                f"Expected '{img_dir}' and '{caption_dir}' folders."
            )

        examples = []
        for caption_path in sorted(caption_dir.glob("*.txt")):
            stem     = caption_path.stem            # '00001' etc.
            img_path = img_dir / f"{stem}.png"
            if not img_path.exists():               # skip unmatched pairs
                continue

            caption = caption_path.read_text(encoding="utf-8").strip()

            examples.append({
                "image":  str(img_path),
                "query":  "Summarize the chart.",          # customise prompt here if needed
                "answer": caption,
                "label":  name.split("/")[0].split("/")[0],
            })
        return examples

    def _to_hf_dataset(self, examples: list[dict]) -> Dataset:
        ds = Dataset.from_list(examples)
        ds = ds.cast_column("image", Image())       # lazy PIL loading
        return ds

    def create_loader(self, data, bsz=1):
        dataloader = DataLoader(data, batch_size = bsz, collate_fn = self.collate_fn_charttotext)
        return dataloader

    def collate_fn_charttotext(self, batch):
        image_batch = []
        query_batch = []
        answer_batch = []
        split_batch = []
        
        for idx in range(len(batch)):
            image_batch.append(batch[idx]['image'])
            query_batch.append(batch[idx]['query'])
            answer_batch.append(batch[idx]['answer'])
            split_batch.append(batch[idx]['label'])

        return image_batch, query_batch, answer_batch, split_batch


