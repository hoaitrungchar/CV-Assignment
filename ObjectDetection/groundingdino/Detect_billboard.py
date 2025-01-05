import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span






def transform_groundingDino():
    return T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model(args,model_config_path, model_checkpoint_path, cpu_only=False):
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, with_logits=True, cpu_only=False, token_spans=None):
    box_threshold=0.3
    text_threshold=0.25
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)



    return boxes_filt, pred_phrases


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
#     parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
#     parser.add_argument(
#         "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
#     )
#     parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
#     parser.add_argument(
#         "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
#     )

#     parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
#     parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
#     parser.add_argument("--token_spans", type=str, default=None, help=
#                         "The positions of start and end positions of phrases of interest. \
#                         For example, a caption is 'a cat and a dog', \
#                         if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
#                         if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
#                         ")

#     parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
#     args = parser.parse_args()

#     # cfg
#     config_file = args.config_file  # change the path of the model config file
#     checkpoint_path = args.checkpoint_path  # change the path of the model
#     image_path = args.image_path
#     text_prompt = args.text_prompt
#     output_dir = args.output_dir
#     box_threshold = args.box_threshold
#     text_threshold = args.text_threshold
#     token_spans = args.token_spans


#     model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)




#     # run model
#     boxes_filt, pred_phrases = get_grounding_output(
#         model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
#     )

