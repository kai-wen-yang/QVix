import torch
from .instruct_blip.models import load_model_and_preprocess
from . import get_image
import pdb


class TestInstructBLIP:
    def __init__(self, args) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna13b", is_eval=True, device=self.device)
        self.temperature = args.temperature

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question}, max_length=max_new_tokens)[0]

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        output = self.model.generate({"image": imgs, "prompt": question_list}, max_length=max_new_tokens,  temperature=self.temperature)
        return output

    @torch.no_grad()
    def text_generate(self, question_list, max_new_tokens=128):
        output = self.model.generate_attention({"prompt": question_list}, device=self.device, max_length=max_new_tokens, temperature=self.temperature)
        return output
