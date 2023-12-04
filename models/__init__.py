import torch
import numpy as np
from PIL import Image

DATA_DIR = '' # Replace it with your model checkpoints save dir

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_BGR_image(image):
    image = get_image(image)
    image = np.array(image)[:, :, ::-1]
    image = Image.fromarray(np.uint8(image))
    return image


def get_model(model_name, args):
    if model_name == 'InstructBLIP':
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP(args)
    elif model_name == 'InstructBLIP7B':
        from .test_instructblip7b import TestInstructBLIP7B
        return TestInstructBLIP7B(args)
    elif model_name == 'LLaVA':
        from .test_llava import TestLLaVA
        return TestLLaVA(args)
    elif model_name == 'MiniGPT':
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(args)
    elif model_name =='Cheetah':
        from .test_cheetah import TestCheetah
        return TestCheetah(args)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
