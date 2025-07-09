# python main.py --mode eval --vlm-name  llava-1.6 --dataset-name chartqa --sft-lora False --dpo-lora False --cot True

# python main.py --mode eval --vlm-name internvl-8b --dataset-name chartqa --cot True

python main.py --mode eval --vlm-name qwen-7b --dataset-name chartqa-src --dpo-lora True #--cot True
# python main.py --mode eval --vlm-name qwen-2b --dataset-name chartqa-src --cot True #--dpo-lora True #--cot True
# python main.py --mode sft --vlm-name internvl-8b --dataset-name chartqa #--cot True

# --dpo-lora True
# --icl True