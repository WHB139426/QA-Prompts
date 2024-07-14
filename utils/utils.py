import torch
from transformers import AutoConfig, StoppingCriteria
import random
import re
import requests
from PIL import Image
from io import BytesIO
from rouge import Rouge
import json
import pickle
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop
from typing import Optional, Tuple, Any, Union, List

def _convert_to_rgb(image):
    return image.convert('RGB')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)
        
def load_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def extract_ans(ans):
        
        answer = ans

        if len(ans) == 1:
            return ans # 'A', 'B', ...
        
        pattern = re.compile(r'\(([A-Z])\)')
        res = pattern.findall(ans)
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
            
        return answer  

def compute_acc(bs, labels, preds):
    acc = 0
    for i in range(bs):
        label = labels[i]
        pred = preds[i]

        if pred.lower() in label.lower():
            acc += 1
            continue

        pred_option = extract_ans(pred)
        label_option = extract_ans(label)

        if len(pred_option) > 1:
            pattern = r'\([A-Z]+\)\s+(.+)'
            match = re.search(pattern, label)
            if match:
                answer = match.group(1).replace(".","")  # 获取第一个捕获组的内容

                if answer.lower() in pred.lower():
                    acc+=1
                    continue
            else:
                pred_option = random.choice(['A','B','C','D'])

        if pred_option == label_option:
            acc+=1
            continue

    return acc/bs

def compute_vqa_score(bs, preds, direct_answers_texts):
    score = 0
    for i in range(bs):
        pred = preds[i].lower()
        direct_answers_text = direct_answers_texts[i].replace("'", '').strip('[]')
        direct_answers_list = [element.lower().strip() for element in direct_answers_text.split(',')]
        freq = direct_answers_list.count(pred)
        score += min(freq/3, 1)
    return score/bs

#######################
# Rouge-L
#######################
def rouge_score(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l

def compute_rouge(bs, labels, preds):
    rouge = 0
    for i in range(bs):
        label = labels[i]
        pred = preds[i]
        if len(pred) == 0:
            pred = 'N/A.'
        rouge += rouge_score(pred, label)
    return rouge/bs

def tagtxt_to_list(tagtxt):
    word_list = tagtxt.split(" | ")
    return word_list

