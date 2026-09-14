import gradio as gr
import sys
from projects.llava_sam2.gradio.app_utils import\
    process_markdown, show_mask_pred, description, preprocess_video,\
    show_mask_pred_video, image2video_and_save, write_mask_frames, process_markdown2, clean_caption

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
import argparse
import os

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args(args):
    parser = argparse.ArgumentParser(description="SG-FSCFormer Demo")
    parser.add_argument('hf_path', help='SG-FSCFormer HuggingFace checkpoint path.')
    return parser.parse_args(args)

def inference(image, video, follow_up, input_str):
    input_image = image
    if image is not None and (video is not None and os.path.exists(video)):
        return image, video, "Error: Please only input a image or a video !!!"
    if image is None and (video is None or not os.path.exists(video)) and not follow_up:
        return image, video, "Error: Please input a image or a video !!!"

    if not follow_up:
        # reset
        print('Log: History responses have been removed!')
        global_infos.n_turn = 0
        global_infos.inputs = ''
        text = input_str

        image = input_image
        global_infos.image_for_show = image
        global_infos.image = image
        video = video
        global_infos.video = video

        if image is not None:
            global_infos.input_type = "image"
        else:
            global_infos.input_type = "video"

    else:
        text = input_str
        image = global_infos.image
        video = global_infos.video

    input_type = global_infos.input_type
    if input_type == "video":
        video = preprocess_video(video, global_infos.inputs+input_str)

    past_text = global_infos.inputs

    if past_text == "" and "<image>" not in text:
        text = "<image>" + text
    if input_type == "image":
        input_dict = {
            'image': image,
            'text': text,
            'past_text': past_text,
            'mask_prompts': None,
            'tokenizer': tokenizer,
        }
    else:
        input_dict = {
            'video': video,
            'text': text,
            'past_text': past_text,
            'mask_prompts': None,
            'tokenizer': tokenizer,
        }

    return_dict = sg_fscformer_model.predict_forward(**input_dict)
    if "past_text" in return_dict:
        global_infos.inputs = return_dict["past_text"]
        print(return_dict['past_text'])
    if 'prediction_masks' in return_dict.keys() and return_dict['prediction_masks'] and len(
            return_dict['prediction_masks']) != 0:
        if input_type == "image":
            image_mask_show, selected_colors = show_mask_pred(global_infos.image_for_show, return_dict['prediction_masks'],)
            video_mask_show = global_infos.video
        else:
            image_mask_show = None
            video_mask_show, selected_colors = show_mask_pred_video(video, return_dict['prediction_masks'],)
            write_mask_frames(video_mask_show, "./vis_output/")
            video_mask_show = image2video_and_save(video_mask_show, save_path="./vis_output/img_frames/ret_video.mp4")
    else:
        image_mask_show = global_infos.image_for_show
        video_mask_show = global_infos.video
        selected_colors = []
    
    predict = return_dict['prediction'].strip()
    global_infos.n_turn += 1
    clean_text_output, pred_phrases = clean_caption(predict)
    
    predict = process_markdown(predict, selected_colors)
    # predict = process_markdown2(predict, selected_colors)
    with open(os.path.join("./vis_output/img_frames/", "caption.txt"), "w") as f:
            f.write(clean_text_output)
    with open(os.path.join("./vis_output/img_frames/", "markdown_caption.md"), "w") as f:
            f.write(predict)
    return image_mask_show, video_mask_show, predict

def init_models(args):
    model_path = args.hf_path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    return model, tokenizer

class global_infos:
    inputs = ''
    n_turn = 0
    image_width = 0
    image_height = 0

    image_for_show = None
    image = None
    video = None

    input_type = "image" # "image" or "video"

if __name__ == "__main__":
    # get parse args and set models
    args = parse_args(sys.argv[1:])

    sg_fscformer_model, tokenizer = \
        init_models(args)

    demo = gr.Interface(
        inference,
        inputs=[
            gr.Image(type="pil", label="Upload Image", height=360),
            gr.Video(sources=["upload", "webcam"], label="Upload mp4 video", height=360),
            gr.Checkbox(label="Follow up Question"),
            gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),],
        outputs=[
            gr.Image(type="pil", label="Output Image"),
            gr.Video(label="Output Video", show_download_button=True, format='mp4'),
            gr.Markdown()],
        theme=gr.themes.Soft(), allow_flagging="auto", description=description,
        title='SG-FSCFormer'
    )

    demo.queue()
    demo.launch(share=True)
    # demo.launch(server_name='115.157.197.138', server_port=9899, share=True)
