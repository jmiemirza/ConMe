
from dec_vl_eval.src.minigpt4.common.config import Config
from dec_vl_eval.src.minigpt4.common.registry import registry
from dec_vl_eval.src.minigpt4.conversation.conversation import Chat, BASE_CONV


def get_chat(args):
    print('Initialization Start')
    cfg = Config(args)

    if args.language_model == 'vicuna_7B':
        cfg.model_cfg.llama_model = "/dccstor/leonidka1/victor_space/weights/full_vicuna_weights/full_vicuna_7B"
        cfg.model_cfg.ckpt = "/dccstor/leonidka1/victor_space/weights/pretrained-mini-gpt4/pretrained_7B.pth"
    elif args.language_model == 'vicuna_13B':
        cfg.model_cfg.llama_model = "/dccstor/leonidka1/victor_space/weights/full_vicuna_weights/full_vicuna_13B"
        cfg.model_cfg.ckpt = "/dccstor/leonidka1/victor_space/weights/pretrained-mini-gpt4/pretrained_13B.pth"
    else:
        raise ValueError("Not a valid language model.")

    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)  # image preprocess transforms: Resize, ToTensor, Normalize
    chat = Chat(model, vis_processor, BASE_CONV, device='cuda:0')

    print('Initialization Finished')
    return chat, cfg
