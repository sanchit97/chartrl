import os
import tqdm
import re
import time
import pickle


import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
from qwen_vl_utils import process_vision_info

import wandb
from models import load_vlm_model
from dataset_process import ChartDataset


from metrics import exact_match, relaxed_accuracy
from utils import get_vlm_output, clear_memory, format_data

import logging, os


# torch.seed_all(2025)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def main():
    mode = "dpo"  # Change to "sft" for supervised fine-tuning
    # Parse command line arguments
    # args = parse_args()
    
    # Load the model and processor
    if mode == "eval":
        # model, processor = load_vlm_model("llava-1.6")
        # model, processor = load_vlm_model("unichart-chartqa")
        model, processor = load_vlm_model("qwen-7b")
        # adapter_path = "llava-1.6-train-chartqa"
        # adapter_path = "qwen-7b-train-chartqa-dpo-v2/checkpoint-775"
        # model = PeftModel.from_pretrained(model, adapter_path)
        # model = model.merge_and_unload() 
        # model, processor = load_vlm_model("internvl-8b")
        dataset = ChartDataset("chartqa", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")
        # test_dataset = dataset.load_chart_dataset(split = "train")
        em_correct, ra_correct, total = 0, 0, 0

        pref_data = []

        for batch in tqdm.tqdm(test_dataset):
            pred = get_vlm_output(model, processor, [batch['image']], [batch['query']])
            print(pred, batch['label'][0])

            # Exact match
            em_correct += exact_match(pred, batch['label'][0])
            # Realaxed accuracy
            ra_correct += relaxed_accuracy([pred],batch['label'])[0]
            if relaxed_accuracy([pred],batch['label'])[0] == 0:
                pref_data.append([batch, pred])
            print(len(pref_data))
                
            #ROUGE-L
            #F1
            #BLEU

            total+=1

            print("EM: ",em_correct/total)
            print("Relaxed Accuracy: ",ra_correct/total)

        # with open('./pref-data/base-llava-1.6-sft', 'wb') as f:
        #     pickle.dump(pref_data, f)


    if mode == "sft":
        os.environ["WANDB_CONSOLE"] = "wrap" 
        wandb.init(project="chartrl", entity="sanchit97")
        # Load the dataset and model for SFT
        # model_name = "llava-1.6"  # Change to the desired model name
        # model_name = "qwen-7b"
        # model_name = "internvl-8b"
        model, processor = load_vlm_model(model_name)
        dataset = ChartDataset("chartqa", processor=processor)
        train_dataset = dataset.load_chart_dataset(split = "train")
        eval_dataset = dataset.load_chart_dataset(split = "val")
        # train_dataset = [format_data(sample) for sample in train_dataset]
        # eval_dataset = [format_data(sample) for sample in eval_dataset]

        logging.info("Loaded model, train and eval datasets")
        logging.info("Using:", model.config._attn_implementation)

        bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_use_double_quant=True, 
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_compute_dtype=torch.bfloat16)
                    
        peft_config = LoraConfig(
                    lora_alpha=16,
                    lora_dropout=0.05,
                    r=8,
                    bias="none",
                    target_modules=["q_proj", "v_proj"],
                    # target_modules = ["wqkv", "wo"],
                    task_type="CAUSAL_LM",)


        peft_model = get_peft_model(model, peft_config)
        print(peft_model.print_trainable_parameters())

        logging.info("Loaded model+LORA adapters")

        training_args = SFTConfig(
            output_dir=model_name+"-train-chartqa-v3",  # Directory to save the model
            num_train_epochs=3,  # Number of training epochs
            per_device_train_batch_size=4,  # Batch size for training
            per_device_eval_batch_size=4,  # Batch size for evaluation
            gradient_accumulation_steps=4,  # Steps to accumulate gradients
            gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
            # Optimizer and scheduler settings
            optim="adamw_torch_fused",  # Optimizer type
            learning_rate=2e-4,  # Learning rate for training
            lr_scheduler_type="constant",  # Type of learning rate scheduler
            # Logging and evaluation
            logging_steps=50,  # Steps interval for logging
            eval_steps=500,  # Steps interval for evaluation
            eval_strategy="steps",  # Strategy for evaluation
            save_strategy="steps",  # Strategy for saving the model
            save_steps=500,  # Steps interval for saving
            metric_for_best_model="eval_loss",  # Metric to evaluate the best model
            greater_is_better=False,  # Whether higher metric values are better
            load_best_model_at_end=True,  # Load the best model after training
            # Mixed precision and gradient settings
            bf16=True,  # Use bfloat16 precision
            tf32=True,  # Use TensorFloat-32 precision
            max_grad_norm=0.3,  # Maximum norm for gradient clipping
            warmup_ratio=0.03,  # Ratio of total steps for warmup
            # Hub and reporting
            push_to_hub=False,  # Whether to push model to Hugging Face Hub
            report_to="wandb" ,  # Reporting tool for tracking metrics
            # Gradient checkpointing settings
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
            # Dataset configuration
            dataset_text_field="",  # Text field in dataset
            dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
            # max_seq_length=1024  # Maximum sequence length for input
        )

        training_args.remove_unused_columns = False  # Keep unused columns in dataset
        training_args.dataloader_num_workers = 4  # Number of workers for data loading

        logging.info("Training arguments set up")

        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=dataset.train_collate_fn_chartqa,
            peft_config=peft_config,
            tokenizer=processor.tokenizer,
        )

        logging.info("SFT Trainer initialized")
        trainer.train()
        logging.info("Training completed, saving output model")
        trainer.save_model(training_args.output_dir)
        logging.info("Model saved to {}".format(training_args.output_dir))


    if mode == "dpo":
        os.environ["WANDB_CONSOLE"] = "wrap" 
        wandb.init(project="chartrl", entity="chartrl")
        # Load the dataset and model for SFT
        model_name = "llava-1.6"  # Change to the desired model name
        # model_name = "qwen-7b"
        model, processor = load_vlm_model(model_name)
        adapter_path = "llava-1.6-train-chartqa"
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload() 
        # breakpoint()
        dataset = ChartDataset("chartqa", processor=processor)
        pref_dataset = dataset.load_pref_data()
        # pref_dataset = pref_dataset.map(dataset.pref_format, batched=True, batch_size=32, num_proc=4)
        # train_dataset = dataset.load_chart_dataset(split = "train")
        eval_dataset = dataset.load_chart_dataset(split = "val")
        logging.info("Loaded model, pref and eval datasets")

        bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_use_double_quant=True, 
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_compute_dtype=torch.bfloat16)
                    
        peft_config = LoraConfig(
                    lora_alpha=16,
                    lora_dropout=0.05,
                    r=8,
                    bias="none",
                    target_modules=["q_proj", "v_proj"],
                    # target_modules = ["wqkv", "wo"],
                    task_type="CAUSAL_LM",)


        peft_model = get_peft_model(model, peft_config)
        print(peft_model.print_trainable_parameters())

        logging.info("Loaded model+LORA adapters")

        training_args = DPOConfig(
        output_dir=model_name+"-train-chartqa-dpo-sft-v1",  # Directory to save the model
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=5,
        dataset_num_proc=32,  # tokenization will use 32 processes
        dataloader_num_workers=32,  # data loading will use 32 workers
        logging_steps=10,
        )
        trainer = DPOTrainer(
            peft_model,
            ref_model=None,  # not needed when using peft
            args=training_args,
            train_dataset=pref_dataset,
            tokenizer=processor,
            peft_config=peft_config,
        )

        trainer.train()

main()






# def parse_args():
#     parser = argparse.ArgumentParser(description="Argument parser for VLM/LLM evaluation pipeline")

#     parser.add_argument('--mode', type=str, choices=["eval", "sft", "ppo", "grpo"], required=True, help='Run mode')

#     parser.add_argument('--vlm-name', type=str, required=False, help='Name of the vision-language model')
#     # parser.add_argument('--llm-name', type=str, required=False, default="llama-3.1-8b", help='Name of the language model')
#     parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset to use')
#     parser.add_argument('--dataset-split', type=str, required=False, default = "test", help='Dataset split')
#     parser.add_argument('--dataset-num-samples', type=int, required=False, help='Number of samples from the dataset')
#     parser.add_argument('--seed', type=int, default=2025, help='Random seed')

    # return parser.parse_args()