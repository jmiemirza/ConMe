import torch
from PIL import Image
import re
pattern = re.compile(r'[A-D]')

def get_model(args, device):
    if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
        from dec_vl_eval.src.llava.model.builder import load_pretrained_model
        from dec_vl_eval.src.llava.utils import disable_torch_init
        from dec_vl_eval.src.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
            KeywordsStoppingCriteria

        disable_torch_init()
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        return tokenizer, model, image_processor
    elif 'minigpt4v2' in args.model:
        from dec_vl_eval.src.minigpt4.common.config import Config
        from dec_vl_eval.src.minigpt4.common.registry import registry
        from dec_vl_eval.src.minigpt4.conversation.conversation import Chat


        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        return chat, model, vis_processor
    else:

        # from lavis.models import load_model_and_preprocess
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, \
            Blip2ForConditionalGeneration, Blip2Processor, AutoTokenizer, AutoModel
            # AutoModelForVision2Seq, IdeficsForVisionText2Text, AutoProcessor


        if args.backend == "lavis":
            if args.model == "flan_t5":
                model_name = "blip2_t5_instruct"
                model_type = "flant5xl"
            elif args.model == "vicuna_7b":
                model_name = "blip2_vicuna_instruct"
                model_type = "vicuna7b"
            elif args.model == "vicuna_13b":
                model_name = "blip2_vicuna_instruct"
                model_type = "vicuna13b"
            else:
                raise ValueError("Model not recognized.")
            # Get the model
            model, vis_processor, _ = load_model_and_preprocess(
                name=model_name,
                model_type=model_type,
                is_eval=True,
                device=device
            )
        elif args.backend == "hf":  #  we focus on HF models for now
            if args.model == "instructblip_flan_t5":
                # model_path = '/system/user/publicdata/llm/InstructBLIP_weights_HF/instructblip-flan-t5-xl/'
                model_path = "Salesforce/instructblip-flan-t5-xl"
                model = InstructBlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                vis_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            elif args.model == "instructblip_vicuna_7b":
                # model_path = '/system/user/publicdata/llm/InstructBLIP_weights_HF/instructblip-vicuna-7b/'
                model_path = "Salesforce/instructblip-vicuna-7b"
                model = InstructBlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                vis_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            elif args.model == "instructblip_vicuna_13b":
                model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b")
                vis_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
            elif args.model == "blip2_coco":
                model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
                vis_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
            elif args.model == "blip2_flan_t5_xl":
                model_path =  '/data2/lin/data/LLM_weights/BLIP2_weights_HF/blip2-flan-t5-xl/'
                model = Blip2ForConditionalGeneration.from_pretrained(model_path)
                vis_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
            elif args.model == 'internlm_xcomposer2_vl_7b':
                tgt_dir = 'internlm/internlm-xcomposer2-vl-7b'
                tokenizer = AutoTokenizer.from_pretrained(tgt_dir, trust_remote_code=True)
                model = AutoModel.from_pretrained(tgt_dir, trust_remote_code=True)
                model.tokenizer = tokenizer
                vis_processor = tokenizer
            elif args.model == 'idefics-9b':
                checkpoint = "HuggingFaceM4/idefics-9b-instruct"
                model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
                vis_processor = AutoProcessor.from_pretrained(checkpoint)
            elif args.model == 'idefics2-8b':
                checkpoint = "HuggingFaceM4/idefics2-8b"
                model = AutoModelForVision2Seq.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
                vis_processor = AutoProcessor.from_pretrained(checkpoint)
            else:
                raise ValueError("Model not recognized.")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

        else:
            raise ValueError("Backend not recognized.")

        return model, vis_processor

def internlm_gen( model, text, images, need_bos=True):
    pt1 = 0
    embeds = []
    im_mask = []
    images = [images]
    images_loc = [0]
    for i, pts in enumerate(images_loc + [len(text)]):
        subtext = text[pt1:pts]
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            need_bos = False
        if i < len(images):
            image = Image.open(images[i]).convert('RGB')
            image = model.vis_processor(image).unsqueeze(0).cuda()
            image_embeds = model.encode_img(image)
            embeds.append(image_embeds)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                        temperature=1.0, max_new_tokens=5, num_beams=5,
                        do_sample=False, repetition_penalty=1.0)

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0]
    res = pattern.findall(output_text)
    if len(res) == 0:
        print('Error:', output_text); res = 'E'
    return res[0]

def model_generate(
        args,
        model,
        sample,
        vis_processor,
        gen_settings,
        device,
        multiple_choices=None
):
    if args.backend == "lavis":
        if 'instructblip' in args.model:
            image = vis_processor["eval"](sample["image"]).unsqueeze(0).to(device)
            sample["image"] = image

            generated_text = model.generate(
                sample,
                **gen_settings
            )[0]
        else:
            raise Exception("Not implemented")

    elif args.backend == "hf":
        letters_2options = ['A. ', 'B. ']

        if 'instructblip' in args.model:
            if multiple_choices is None or len(multiple_choices) == 0:
                input_text = sample["prompt"] + '\nAnswer:'
            else:
                input_text = sample["prompt"] + ("Answer with the option's letter, A or B, from the given choices below." if 'instructblip_vicuna_7b' in args.model else "") + '\n' + '\n'.join(
                    [letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
                ) + '\n' + ("Answer with the option's letter from the given choices directly. Answer:" if 'instructblip_vicuna_7b' not in args.model else "Answer:")

            inputs = vis_processor(
                images=sample["image"],
                text=input_text,
                return_tensors="pt",
                max_length=512
            ).to(device, torch.float16)

            outputs = model.generate(
                **inputs,  # inputs.data  is dict of   'inputs_ids', 'attention_mask', 'qformer_input_ids', 'qformer_attention_mask', 'pixel_values'
                **gen_settings
            )  #  outputs are integer tokens
            generated_text = vis_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        elif 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
            pass
        elif 'internlm' in args.model:
            options_prompt = f'A. {multiple_choices[0]} B. {multiple_choices[1]} '
            img_prompt = '[UNUSED_TOKEN_146]user\n'
            context = 'N/A'
            options_prompt = options_prompt.strip()
            mid_prompt = 'Question: ' + sample['prompt'] + '\nContext: ' + context + '\nOptions: ' + options_prompt
            ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'
            full_instruction_prompt = img_prompt + mid_prompt + ans_prompt

            generated_text = internlm_gen(model, full_instruction_prompt, sample["image_path"])
        elif 'idefics-9b' in args.model:
            prompts = [
                "User:",
                sample["image"],
                f'{sample["prompt"]}' + '\n' + '\n'.join(
                        [letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
                    ) + '\n' + "Answer with the option's letter from the given choices directly.\nAssistant:"
            ]
            inputs = vis_processor(prompts, return_tensors='pt').to(device)
            generated_ids = model.generate(**inputs, **gen_settings)
            generated_text = vis_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        elif 'idefics2-8b' in args.model:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f'{sample["prompt"]}' + '\n' + '\n'.join(
                                [letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
                            ) + '\n' + "Answer with the option's letter from the given choices directly. Answer:"},
                    ]
                },
            ]
            message_text = vis_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = vis_processor(text=message_text, images=[sample['image']], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated_ids = model.generate(**inputs, **gen_settings)
            generated_text = vis_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()[-1]

        else:
            raise Exception("Not implemented")
    else:
        raise ValueError("Backend not implemented")

    return generated_text

def t2i_score(
        sample,
        multiple_choices,
        correct_choice,
        t2i_model
):
    scores = dict()
    for choice in multiple_choices:
        score = t2i_model(images=[sample['image_path']], texts=[choice])
        scores[choice] = score.item()

    choice = max(scores, key=lambda key: scores[key])  # compare the scores of the two captions
    hit = int(choice == correct_choice)

    return hit, scores
