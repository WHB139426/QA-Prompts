import torch
import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from utils import *
from utils.utils import *
from finetune_ans import prepare_inputs
from datasets.aokvqa_dataset import AOKVQADataset
from models.blip2_vicuna_instruct import Blip2VicunaInstruct

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='experiments/vicuna_1_0.6969.pth')
parser.add_argument('--coco_path', type=str, default="../coco2017")
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='aokvqa')
parser.add_argument('--use_qaprompt', type=bool, default=True)
parser.add_argument('--eval_bs', type=int, default=10)
parser.add_argument('--model', type=str, default='instruct_blip', choices=['instruct_blip'])
args = parser.parse_args()

generate_kwargs = {
    "do_sample": True,
    "num_beams": 5, 
    "min_length": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 10,
    "temperature":0.7,
    "top_p":0.9,
    }

model = Blip2VicunaInstruct(dtype=torch.float16, use_qaprompt = True)
model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
model.to(args.device)
model.eval()

val_dataset = AOKVQADataset(anno_path = "annotations/aokvqa_v1p0_val.json", subqa_path = 'annotations/sub_questions.json', img_path = os.path.join(args.coco_path, 'val2017'))
val_loader = DataLoader(val_dataset, batch_size=args.eval_bs, shuffle=False, drop_last=False, num_workers=4)

val_vqa_score = 0
for step, data in enumerate(tqdm(val_loader)):
    text_input, text_output, questions, concat_subqa_texts, direct_answers_texts = prepare_inputs(args, data)
    samples = {
            "text_input": text_input,
            "text_output": text_output,
            "questions": questions,
            "qaprompts": concat_subqa_texts,
            "pixel_values": data["pixel_values"].to(args.device),
        }
    with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype): # 前后开启autocast
        with torch.no_grad():
            pred_texts = model.generate(samples, **generate_kwargs)
    val_vqa_score += compute_vqa_score(bs=len(pred_texts), direct_answers_texts = direct_answers_texts, preds = pred_texts)

val_vqa_score = round(val_vqa_score/len(val_loader), 4)

print("val_vqa_score: ", val_vqa_score)
