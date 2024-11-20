# Q&A Prompts-ECCV'24

**Q&A Prompts: Discovering Rich Visual Clues through Mining Question-Answer Prompts for VQA requiring Diverse World Knowledge.** This is the official implementation of the [[Paper](https://arxiv.org/abs/2401.10712)] accepted by ECCV'24.

## Install

1. Clone this repository and navigate to QA-Prompts folder
```bash
git clone https://github.com/WHB139426/QA-Prompts.git
cd QA-Prompts
mkdir experiments
```

2. Install Package
```Shell
conda create -n qaprompts python=3.9.16
conda activate qaprompts
pip install -r requirements.txt
pip install numpy==1.26.4
```

## Datasets

We prepare the annotations of [[A-OKVQA](https://allenai.org/project/a-okvqa/home)] in `./annotations`. 

The images can be downloaded from [[COCO2017](https://cocodataset.org/#download)], and you should organize the data as follows,

```
├── coco2017
│   └── train2017
│   └── val2017
│   └── test2017
├── QA-Prompts
│   └── annotations
│     └── aokvqa_v1p0_train.json
│     └── sub_qa.json
│     └── ...
│   └── datasets
│   └── models
│   └──...
```
You should also modify the parameter `coco_path` of argparse in `finetune_ans.py`/`evaluation.py` according to the directory of your COCO images.

## Pretrained Weights of InstructBLIP

You can prepare the pretrained weights of InstructBLIP-Vicuna-7B according to [[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)].

Since we have changed the structure of the code of the model, we RECOMMEND you download the pretrained weights of EVA-CLIP, Vicuna-7b-v1.1 and QFormer directly in [[🤗HF](https://huggingface.co/WHB139426/QAprompts/tree/main)]. The pretrained weights should be organized as follows,

```
├── QA-Prompts
│   └── experiments
│     └── eva_vit_g.pth
│     └── qformer_vicuna.pth
│     └── query_tokens_vicuna.pth
│     └── vicuna-7b
│     └── llm_proj_vicuna.pth
```

## Evaluation

Download the trained checkpoints `vicuna_1_0.6969.pth` from [[🤗HF](https://huggingface.co/WHB139426/QAprompts/tree/main)] (should be stored in `./experiments`), and then run

```Shell
python evaluation.py
```

## Training

We recommend using GPUs with memory > 24G. Otherwise, you may need to extract the vision features in advance to save the memory usage of EVA-CLIP and avoid OOM. Modify the parameter `world_size` of argparse in `finetune_ans.py` according to the number of GPUs.

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1111 finetune_ans.py
```


