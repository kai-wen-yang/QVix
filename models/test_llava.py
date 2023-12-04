import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import pdb


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


class TestLLaVA:
    def __init__(self, args):
        disable_torch_init()
        # model_path = "liuhaotian/llava-v1.5-7b"
        # model_name = get_model_name_from_path(model_path)
        # self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        #      model_path, None, model_name)
        model_path = "liuhaotian/LLaVA-Lightning-MPT-7B-preview"
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
             model_path,None, model_name)
        # model_path = "liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview"
        # model_name = get_model_name_from_path(model_path)
        # self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        #      model_path, '/fs/nexus-scratch/kwyang3/models/lamma2', model_name)

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        self.conv = conv_templates[conv_mode].copy()
        self.image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        self.temperature = args.temperature


    def batch_generate(self, images, questions, max_new_tokens=128):
        outputs = []
        for image, qs in zip(images, questions):
            if self.model.config.mm_use_im_start_end:
                qs = self.image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = self.conv.copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = load_image(image)
            image_tensor = process_images(
                [image],
                self.image_processor,
                self.model.config
            ).to(self.model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            )
        
            stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=self.temperature,
                    max_length=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )
            output = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            output = output.strip()
            if output.endswith(stop_str):
                output = output[: -len(stop_str)]
            output = output.strip()
            outputs.append(output)
        return outputs
            

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        vision_tower = self.model.get_model().get_vision_tower()#.vision_tower[0]
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)
    
