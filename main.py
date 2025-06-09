import os
import tqdm
import re

import torch


from qwen_vl_utils import process_vision_info

from models import load_vlm_model
from dataset_process import load_chart_dataset


def get_prompt_temp(q):
    prefix = "Look at the chart. %s Answer with a single phrase only."%q
    return prefix

def normalize_answer(ans):
    ans = str(ans)
    # Keep only alphanumerics and spaces
    ans = re.sub(r'[^A-Za-z0-9 .]+', '', ans)
    return ans.strip().lower()


def get_vlm_output(model, processor, image, query):
    if "llava" in model.__class__.__name__.lower():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": get_prompt_temp(query[0])},
                    {"type": "image"},
                ]
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image[0], text=prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=100)
        out_text = processor.decode(output[0], skip_special_tokens=True)
        out_text = out_text.split("[/INST]")[-1].strip()


    elif "qwen" in model.__class__.__name__.lower():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": get_prompt_temp(query[0])},
                    {
                        "type": "image",
                        "image": image[0],
                    },
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        out_text = output_text[0]
    

    elif "internvl" in model.__class__.__name__.lower():
        from internvl_utils import build_transform, find_closest_aspect_ratio, dynamic_preprocess, load_image, get_conv_template
        generation_config = {"num_beams":1,"max_new_tokens":100}
        question =  "<image> "+ get_prompt_temp(query[0])
        pixel_values = load_image(image[0])
        history=None 
        return_history=False
        num_patches_list=None 
        IMG_START_TOKEN='<img>' 
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id


        template = get_conv_template(model.template)
        template.system_message = model.system_message
        eos_token_id = processor.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()


        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = processor(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id

        
        outputs = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )

        out_text = processor.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    return normalize_answer(out_text)


def exact_match(pred, label):
    if pred.lower() == label.lower():
        return 1
    return 0

def main():
    # model, processor = load_vlm_model("llava-1.6")
    # model, processor = load_vlm_model("qwen-7b")
    model, processor = load_vlm_model("internvl-8b")
    loader = load_chart_dataset("chartqa")

    em_correct, total = 0, 0

    for batch in tqdm.tqdm(loader):
        pred = get_vlm_output(model, processor, batch[0], batch[1])
        print(pred, batch[2][0][0])

        # Exact match
        em_correct += exact_match(pred, batch[2][0][0])
        #ROUGE-L
        #F1
        #BLEU

        total+=1

        print("Correct: ",em_correct/total)

        print()

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

#     return parser.parse_args()