import torch
from omegaconf import OmegaConf
import cheetah
from cheetah.common.config import Config
from cheetah.common.registry import registry
from cheetah.conversation.conversation import Chat, CONV_VISION

from cheetah.models import *
from cheetah.processors import *
from . import get_image
import pdb


class TestCheetah:
    def __init__(self, args) -> None:
        config = OmegaConf.load("/fs/nexus-scratch/kwyang3/good_question/models/cheetah/cheetah_eval_vicuna.yaml")
        cfg = Config.build_model_config(config)
        model_cls = registry.get_model_class(cfg.model.arch)
        self.model = model_cls.from_config(cfg.model).cuda()
        vis_processor_cfg = cfg.preprocess.vis_processor.eval
        self.vis_processors = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.chat = Chat(self.model, self.vis_processors, device=self.device)


    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question}, max_length=max_new_tokens)[0]

        return output

    @torch.no_grad()
    def condition_batch_generate(self, image_list, question_list, max_new_tokens=128):
        raw_img_list = []
        for image in image_list:
            raw_img_list.append([image])

        context = ["<ImageHere> Where and when is this picture taken?" for _ in question_list]
        outputs = self.chat.batch_answer(raw_img_list, context)

        for i in range(len(context)):
            context[i] += (" " + CONV_VISION.sep + CONV_VISION.roles[1] + ": " + outputs[i] \
                           + " " + CONV_VISION.sep+CONV_VISION.roles[0] + ": " + \
                           "What is the shape and color of the object in the picture? ")

        outputs = self.chat.batch_answer(raw_img_list, context)
        for i in range(len(context)):
            context[i] += (" " + CONV_VISION.sep + CONV_VISION.roles[1] + ": " + outputs[i] \
                           + " " + CONV_VISION.sep+CONV_VISION.roles[0] + ": " + \
                           "<ImageHere> What is the object in the picture? ")
        raw_img_list = []
        for image in image_list:
            raw_img_list.append([image, image])
        
        outputs = self.chat.batch_answer(raw_img_list, context, max_new_tokens=max_new_tokens)
        # outputs = []
        # for image, question in zip(image_list, question_list):
        #     output = self.chat.answer([image], "<Img><HereForImage></Img> "+question, max_new_tokens=max_new_tokens)
        #     outputs.append(output)
        return outputs
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        raw_img_list = []
        for image in image_list:
            raw_img_list.append([image])

        outputs = self.chat.batch_answer(raw_img_list, question_list, max_new_tokens=max_new_tokens)

        return outputs


    @torch.no_grad()
    def batch_generate2(self, image_list, question_list, max_new_tokens=128):
        raw_img_list = []
        for image in image_list:
            raw_img_list.append([image, image])

        outputs = self.chat.batch_answer(raw_img_list, question_list, max_new_tokens=max_new_tokens)

        return outputs

    @torch.no_grad()
    def text_generate(self, question_list, max_new_tokens=128,
         use_nucleus_sampling=False,
         num_beams=5,
         device='cpu',
         max_length=256,
         min_length=1,
         top_p=0.9,
         repetition_penalty=1.5,
         length_penalty=1,
         num_captions=1,
         temperature=1,
        ):
        self.model.llama_tokenizer.padding_side = "right"

        llm_tokens = self.model.llama_tokenizer(
            question_list,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)
        pdb.set_trace()
        with self.model.maybe_autocast():
            inputs_embeds = self.model.llama_model.get_input_embeddings()(llm_tokens.input_ids)
            attention_mask = llm_tokens.attention_mask

            outputs = self.model.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text
