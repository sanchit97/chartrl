import os
import tqdm
import re
import time
import gc

import torch
from qwen_vl_utils import process_vision_info


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
        system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
        Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
        The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
        Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
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

    elif 'visionencoderdecodermodel' in model.__class__.__name__.lower():
        input_prompt = '<chartqa> ' + query[0] + ' <s_answer>'
        print(input_prompt)
        decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = processor(image[0].convert("RGB"), return_tensors="pt").pixel_values
        outputs = model.generate(
            pixel_values.to(model.device),
            decoder_input_ids=decoder_input_ids.to(model.device),
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,)

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = sequence.split("<s_answer>")[1].strip()

        out_text = sequence
        
    return normalize_answer(out_text)

def get_prompt_temp(q):
    prefix = "Look at the chart. %s Answer with a single phrase only."%q
    return prefix

def normalize_answer(ans):
    ans = str(ans)
    # Keep only alphanumerics and spaces
    ans = re.sub(r'[^A-Za-z0-9 .]+', '', ans)
    return ans.strip().lower()

def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def format_data(sample):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]