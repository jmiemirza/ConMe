

from PIL import Image

def chat_encode_img(chat, image_path, ):
    raw_image = Image.open(image_path).convert('RGB')
    image = chat.vis_processor(raw_image).unsqueeze(0).cuda()
    img_embed_query = chat.model.encode_img(image)[0]
    return img_embed_query