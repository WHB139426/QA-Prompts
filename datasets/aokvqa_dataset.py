from torch.utils.data import Dataset
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import pickle
import sys
import os
import requests
from PIL import Image
from collections import Counter
from io import BytesIO
import json
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *

def load_aokvqa_annotations(annotations_jsonpath, test=False):
    entries = []
    with open(annotations_jsonpath, 'r', encoding = 'utf-8') as fp:
        aokvqa_data = json.load(fp)
    if test == False:
        for sample in aokvqa_data:
            entries.append(
                {
                    "img_id": str(sample['image_id']).zfill(12), # 000000000139
                    "question_id": sample['question_id'],
                    "split": sample['split'],
                    "question": sample['question'],
                    "choices": sample['choices'],
                    "correct_choice_idx": sample['correct_choice_idx'],
                    "direct_answers": sample['direct_answers'],
                    "rationales": sample['rationales']
                }
            )
    else:
        for sample in aokvqa_data:
            entries.append(
                {
                    "img_id": str(sample['image_id']).zfill(12), # 000000000139
                    "split": sample['split'],
                    "question": sample['question'],
                    "choices": sample['choices'],
                    "question_id": sample['question_id']
                }
            )
    return entries

def search_subqa_item(context_list, qid, img_id):
    for item in context_list:
        if str(item["question_id"]) == str(qid) and str(item["image_id"]) == str(img_id):
            return item['sub_qa']
    return 'N/A.'

def construct_sub_qa_text(sub_qa):
    sub_qa_num = 8
    no_list = ['man', 'person', 'stand', 'sit', 'woman', 'white', 'catch', 'table', 'floor', 'food', 'road', 'ride', 'park', 'walk', 'building', 'play', 'green', 'water', 'red', 'black', 'yellow', 'blue']
    sub_qa = [item for item in sub_qa if item['tag_answer'] not in no_list]
    sub_qa_list = []
    if len(sub_qa) < sub_qa_num:
        for i in range(sub_qa_num):
            k = i % len(sub_qa)
            sub_qa_list.append(f"Question: {sub_qa[k]['sub_question']} Short answer: {sub_qa[k]['tag_answer']}")

    else:
        sub_qa = sub_qa[:sub_qa_num]
        for i in range(len(sub_qa)):
            sub_qa_list.append(f"Question: {sub_qa[i]['sub_question']} Short answer: {sub_qa[i]['tag_answer']}")
    return sub_qa_list

class AOKVQADataset(Dataset):
    def __init__(
        self,
        anno_path = "annotations/aokvqa_v1p0_train.json",
        subqa_path = 'annotations/sub_questions.json',
        img_path = "../coco2017/train2017",
    ):
        self.data = load_aokvqa_annotations(anno_path)
        self.sub_qas = load_json(subqa_path)
        self.image_processor = image_transform(image_size=224)

        self.option_0 = []
        self.option_1 = []
        self.option_2 = []
        self.option_3 = []
        self.questions = []
        self.direct_answers = []
        self.open_answer_texts = []
        self.mc_answer_texts = []
        self.image_ids = []
        self.question_ids = []
        self.img_path = img_path
        self.sub_qa_texts = []

        for data in self.data:
            question_id = data['question_id']
            image_file = data['img_id']
            question = data['question']
            direct_answers = data['direct_answers']
            most_ans = Counter(direct_answers).most_common()[0][0]
            choices = data['choices']
            correct_choice_idx = data['correct_choice_idx']
            sub_qa_text = construct_sub_qa_text(search_subqa_item(self.sub_qas, question_id, image_file))

            self.image_ids.append(image_file)
            self.question_ids.append(question_id)
            self.questions.append(question)
            self.open_answer_texts.append(most_ans) # choices[correct_choice_idx], most_ans
            self.mc_answer_texts.append(choices[correct_choice_idx])
            self.direct_answers.append(direct_answers)
            self.sub_qa_texts.append(sub_qa_text)
            self.option_0.append(choices[0])
            self.option_1.append(choices[1])
            self.option_2.append(choices[2])
            self.option_3.append(choices[3])

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.image_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        sub_qa = self.sub_qa_texts[index]
        question_id = str(self.question_ids[index])
        question = str(self.questions[index])
        option_0 = self.option_0[index]
        option_1 = self.option_1[index]
        option_2 = self.option_2[index]
        option_3 = self.option_3[index]
        direct_answers = str(self.direct_answers[index])
        open_answer_text = str(self.open_answer_texts[index])
        mc_answer_text = str(self.mc_answer_texts[index])
        image_file = self.image_ids[index]
        pixel_values = (self.image_processor(load_image(self.img_path + f"/{image_file}.jpg"))) # [3, 224, 224]


        return {
                "image_ids": image_file,
                "question_ids": question_id,

                "pixel_values": pixel_values,

                "questions": question,
                "option_0": option_0,
                "option_1": option_1,
                "option_2": option_2,
                "option_3": option_3,
                "open_answer_texts": open_answer_text,
                "mc_answer_texts": mc_answer_text,
                "direct_answers_texts": direct_answers,

                "sub_qas": sub_qa,
            }
        
     
    
# train_dataset = AOKVQADataset()
# val_dataset = AOKVQADataset()

# for i in range(10):
#     entry = random.choice(train_dataset)
#     print(entry['question_ids'], entry['image_ids'])
#     print(entry['pixel_values'].shape)
#     print("question: ",             entry['questions'])
#     print("options: ", entry['option_0'], ",", entry['option_1'], ",",  entry['option_2'], ",",  entry['option_3'])
#     print("direct_answers_texts: ", entry['direct_answers_texts'])
#     print("open_answer_texts: ",   entry['open_answer_texts'])
#     print("mc_answer_texts: ",    entry['mc_answer_texts'])
#     print("sub_qas: ",              entry['sub_qas'])
#     print()

# data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, pin_memory=True, shuffle=False, drop_last=True, num_workers=8)
# for step, data in enumerate(data_loader):
#     pixel_values = data["pixel_values"]
#     sub_qas = data["sub_qas"]
#     questions = data['questions']
#     open_answer_texts = data['open_answer_texts']

#     subqa_texts_input = []
#     for j in range(pixel_values.shape[0]):
#         subqa_texts_input.append([sub_qas[i][j] for i in range(len(sub_qas))])
#     concat_subqa_texts = []
#     sep = '\n'
#     for qa_list in subqa_texts_input:
#         txt = ''
#         for qa in qa_list:
#             txt += qa + sep
#         concat_subqa_texts.append(txt)
    
#     for i in range(pixel_values.shape[0]):
#         print(questions[i])
#         print(open_answer_texts[i])
#         print(concat_subqa_texts[i])
#         print()
#     break