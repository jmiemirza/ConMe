
import requests
from PIL import Image
from io import BytesIO

from dec_vl_eval.src.llava.mm_utils import process_images
import torch
str_to_remove = [ ' </s>', '\n', '</s>', '.']

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def compute_image_tensor_llava(file_name, args, model, image_processor):
    # load image
    image = load_image(file_name)
    image_tensor = process_images([image], image_processor, args)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    return image, image_tensor