import numpy as np
import torch
import os
from tqdm import tqdm
import argparse
from torch import cuda
import time
from utils import *
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
import pickle
import random
import json
from torch.backends import cudnn
from utils.utils import *

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --word_size=4 --master_port=1111 finetune_ans.py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, default='experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    parser.add_argument('--coco_path', type=str, default="/home/haibo/data/coco2017")
    parser.add_argument('--word_size', default=4, help="n_gpus")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--eval_bs', type=int, default=1) 
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eval_step', type=int, default=4, help="eval every 1/eval_step epoch")
    parser.add_argument('--dataset', type=str, default='aokvqa', choices=['aokvqa'])
    parser.add_argument('--use_qaprompt', type=bool, default=True)
    parser.add_argument('--model', type=str, default='instruct_blip', choices=['instruct_blip'])

    args = parser.parse_args()
    return args
        
def reduce_metric(metric):
    metric_tensor = torch.tensor(metric).cuda(args.local_rank)
    dist.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
    metric = metric_tensor.item() / dist.get_world_size()
    return metric

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def prepare_inputs(args, data):

    image_ids = data["image_ids"]
    question_ids = data["question_ids"]

    questions = data["questions"] 
    sub_qas = data["sub_qas"]
    direct_answers_texts = data["direct_answers_texts"] if 'direct_answers_texts' in data.keys() else ['N/A' for i in range(len(questions))]

    subqa_texts_input = []
    for j in range(len(questions)):
        subqa_texts_input.append([sub_qas[i][j] for i in range(len(sub_qas))])
    concat_subqa_texts = []
    sep = '\n'
    for qa_list in subqa_texts_input:
        txt = ''
        for qa in qa_list:
            txt += qa + sep
        concat_subqa_texts.append(txt)

    if args.model == 'instruct_blip':
        instruction = ''
        post_prompt = '\nShort answer:'

    if args.dataset == 'aokvqa':
        options_0 = data["option_0"]
        options_1 = data["option_1"]
        options_2 = data["option_2"]
        options_3 = data["option_3"]
        if args.use_qaprompt:
            text_input =  [instruction + 'Question: ' + question + f'\nPossible options: {option_0}, {option_1}, {option_2}, {option_3}' + post_prompt for question, option_0, option_1, option_2, option_3 in zip(questions, options_0, options_1, options_2, options_3)]
        else:
            text_input =  [instruction + 'Question: ' + question + post_prompt for question in questions]
        text_output = data["open_answer_texts"]

    return text_input, text_output, questions, concat_subqa_texts, direct_answers_texts

@torch.no_grad()
def eval(args, val_loader, model):
    model.eval()

    val_loss = 0
    val_vqa_score = 0

    for step, data in enumerate(val_loader):
        text_input, text_output, questions, concat_subqa_texts, direct_answers_texts = prepare_inputs(args, data)
        samples = {
                "text_input": text_input,
                "text_output": text_output,
                "questions": questions,
                "qaprompts": concat_subqa_texts,
                "pixel_values": data["pixel_values"].cuda(args.local_rank),
            }
        generate_kwargs = {
            "do_sample": True,
            "num_beams": 5, 
            "min_length": 1,
            "num_return_sequences": 1,
            "max_new_tokens": 10,
            "temperature":0.7,
            "top_p":0.9,
            }
        with torch.cuda.amp.autocast(enabled=True, dtype=model.module.dtype): # 前后开启autocast
            with torch.no_grad():
                outputs = model(samples)
                pred_texts = model.module.generate(samples, **generate_kwargs)

        loss = outputs['loss']
        val_loss += loss.item() 
        val_vqa_score += compute_vqa_score(bs = args.eval_bs, direct_answers_texts = direct_answers_texts, preds = pred_texts)

        if dist.get_rank() == 0 and step<=25:
            for i in range(len(text_input)):
                print()
                print("---------------------eval-------------------------")
                print("image_ids: " + data["image_ids"][i] + "  question_ids: " + data["question_ids"][i])
                print("---------------------subqa-------------------------")
                print(concat_subqa_texts[i])
                print("---------------------input-------------------------")
                print(text_input[i])
                print("---------------------preds-------------------------")
                print(pred_texts[i])
                print("--------------------answers------------------------")
                print(direct_answers_texts[i])
                print()

    # 对不同进程上的评价指标进行平均
    val_loss = round(reduce_metric(val_loss)/len(val_loader), 4)
    val_vqa_score = round(reduce_metric(val_vqa_score)/len(val_loader), 4)
    model.train()
    return val_loss, val_vqa_score

def train(args, train_dataset, val_dataset, model):

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler, pin_memory=True, shuffle=False, drop_last=True, num_workers=4*args.word_size)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_bs, sampler=val_sampler, pin_memory=True, shuffle=False, drop_last=True, num_workers=4*args.word_size)

    optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr = args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0)

    max_score = 0
    save_socre = 0.75

    scaler = torch.cuda.amp.GradScaler() #训练前实例化一个GradScaler对象

    for epoch in range(args.epoch):
        model.train()
        # 设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
        train_loader.sampler.set_epoch(epoch)
        start = time.time()
        train_loss = 0
        train_vqa_score = 0
        
        for step, data in enumerate(tqdm(train_loader, disable=not dist.get_rank() == 0)):
            text_input, text_output, questions, concat_subqa_texts, direct_answers_texts = prepare_inputs(args, data)
            samples = {
                    "text_input": text_input,
                    "text_output": text_output,
                    "questions": questions,
                    "qaprompts": concat_subqa_texts,
                    "pixel_values": data["pixel_values"].cuda(args.local_rank),
                }

            with torch.cuda.amp.autocast(enabled=True, dtype=model.module.dtype): # 前后开启autocast
                outputs = model(samples)
                with torch.no_grad():
                    pred_texts = ['N/A' for i in range(args.bs)]

            loss = outputs['loss']
            train_loss += loss.item() 
            train_vqa_score += compute_vqa_score(bs = args.bs, direct_answers_texts = direct_answers_texts, preds = pred_texts)

            if step % int(len(train_loader)/args.eval_step) == 0 and step >= int(len(train_loader)/args.eval_step) and step < len(train_loader)*0.9:
                val_loss, val_vqa_score = eval(args, val_loader, model)
                if dist.get_rank() == 0:
                    print('epoch:{}/{} step:{}  val_loss:{}  val_vqa_score:{}'
                        .format(epoch + 1, args.epoch, step, val_loss, val_vqa_score))
                    if (val_vqa_score >= max_score and val_vqa_score>save_socre):    
                        max_score = val_vqa_score
                        torch.save(model.module.state_dict(), './{}/{}_{}_{}.pth'.format(args.experiment_path, 'vicuna', epoch+1, val_vqa_score))

            scaler.scale(loss).backward()  #为了梯度放大

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)#使用第二种裁剪方式。

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 对不同进程上的评价指标进行平均
        train_loss = round(reduce_metric(train_loss)/len(train_loader), 4)
        train_vqa_score = round(reduce_metric(train_vqa_score)/len(train_loader), 4)
        val_loss, val_vqa_score = eval(args, val_loader, model)
        
        end = time.time()
        if dist.get_rank() == 0:
            print('epoch:{}/{}  time:{}h  lr:{}  batchsize:{}  train_loss:{}  val_loss:{}  train_vqa_score:{}  val_vqa_score:{} '
                .format(epoch + 1, args.epoch, str(round((end-start)/3600, 2)), args.lr, args.bs, train_loss, val_loss, train_vqa_score, val_vqa_score))
            if (val_vqa_score >= max_score and val_vqa_score>save_socre): 
                max_score = val_vqa_score
                torch.save(model.module.state_dict(), './{}/{}_{}_{}.pth'.format(args.experiment_path, args.model, epoch+1, val_vqa_score))

    dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()

    from datasets.aokvqa_dataset import AOKVQADataset
    if args.dataset == 'aokvqa':
        train_dataset = AOKVQADataset(anno_path = "annotations/aokvqa_v1p0_train.json", subqa_path = 'annotations/sub_questions.json', img_path = os.path.join(args.coco_path, 'train2017'))
        val_dataset = AOKVQADataset(anno_path = "annotations/aokvqa_v1p0_val.json", subqa_path = 'annotations/sub_questions.json', img_path = os.path.join(args.coco_path, 'val2017'))

    if args.model == 'instruct_blip':
        from models.blip2_vicuna_instruct import Blip2VicunaInstruct
        model = Blip2VicunaInstruct(
                dtype=torch.float16,
                use_qaprompt = args.use_qaprompt
            )

    device = torch.device('cuda', args.local_rank)
    dist.init_process_group(backend='nccl',rank=args.local_rank, world_size=args.word_size)
    init_seeds(args.seed + torch.distributed.get_rank())
    torch.cuda.set_device(device)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(args.local_rank),
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank, # find_unused_parameters=True
                                                        ) # ,find_unused_parameters=True # broadcast_buffers=False
    if dist.get_rank() == 0:
        print(get_parameter_number(model))
        print("dataset: {}  train_num: {}  eval_num: {}  epochs: {}  batch_size_per_gpu: {}  n_gpu: {}  learning_rate: {}".format(args.dataset, len(train_dataset), len(val_dataset), args.epoch, args.bs, args.word_size, args.lr))

    train(args, train_dataset, val_dataset, model)
