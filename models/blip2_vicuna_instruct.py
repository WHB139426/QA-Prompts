import sys
import os
import contextlib
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from transformers import BertTokenizer, LlamaTokenizer
from transformers import BertConfig, Blip2Config
from transformers import LlamaForCausalLM, Blip2VisionModel
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *
from models.Qformer import BertLMHeadModel
from models.prompt_module import QAPrompting
os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
class Blip2VicunaInstruct(nn.Module):
    def __init__(
        self,
        dtype=torch.float16,
        use_qaprompt = True
    ):
        super().__init__()
        self.dtype = dtype
        self.max_input_txt_len = 256
        self.max_output_txt_len = 256
        self.use_qaprompt = use_qaprompt
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        print('loading ViT')
        blip2_config = Blip2Config.from_pretrained('Salesforce/blip2-flan-t5-xl')
        blip2_config.vision_config.torch_dtype = self.dtype
        self.vision_model = Blip2VisionModel(blip2_config.vision_config)
        self.vision_model.load_state_dict(torch.load("./experiments/eva_vit_g.pth", map_location='cpu'))

        print('loading Qformer')
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token=32, vision_width=blip2_config.vision_config.hidden_size, cross_attention_freq=2)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        self.Qformer.load_state_dict(torch.load("./experiments/qformer_vicuna.pth", map_location='cpu'))
        self.query_tokens = nn.Parameter(torch.load("./experiments/query_tokens_vicuna.pth", map_location='cpu'))

        print('loading Vicuna')
        self.llm_tokenizer = LlamaTokenizer.from_pretrained('./experiments/vicuna-7b', use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained('./experiments/vicuna-7b', torch_dtype=self.dtype)
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        print('loading llm_proj')
        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)
        self.llm_proj.load_state_dict(torch.load("./experiments/llm_proj_vicuna.pth", map_location='cpu'))

        if self.use_qaprompt:
            print("loading QAPrompting")
            self.QAprompting_module = QAPrompting()

        print("Frozen ViT")
        for name, param in self.vision_model.named_parameters():
            param.requires_grad = False
        self.vision_model = self.vision_model.eval()

        print("Frozen vicuna")
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        self.llm_model = self.llm_model.eval()

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def init_tokenizer(self, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.torch_dtype = self.dtype
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def encode(self, samples):
        text_input = samples["text_input"]
        pixel_values = samples["pixel_values"]

        with self.maybe_autocast():
            image_embeds = self.vision_model(pixel_values=pixel_values)[0]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        text_Qformer = self.tokenizer(
            samples['questions'],
            padding='longest',
            truncation=True,
            max_length=self.max_input_txt_len,
            return_tensors="pt",
        ).to(self.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(self.device)

        if self.use_qaprompt:
            prompt_tokens = self.QAprompting_module(image_embeds, samples['qaprompts'])
            prompt_atts = torch.ones(prompt_tokens.size()[:-1], dtype=torch.long).to(self.device)
            inputs_llm = torch.cat([inputs_llm, prompt_tokens], dim=1)
            atts_llm = torch.cat([atts_llm, prompt_atts],dim=1)

        return inputs_llm, atts_llm
    
    def forward(self, samples):

        inputs_llm, atts_llm = self.encode(samples)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_input_txt_len,
        ).to(inputs_llm.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(inputs_llm.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        with self.maybe_autocast():
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(inputs_llm.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        **generate_kwargs
    ):
        
        inputs_llm, atts_llm = self.encode(samples)

        self.llm_tokenizer.padding_side = "left"

        llm_tokens = self.llm_tokenizer(
            samples["text_input"],
            padding="longest",
            return_tensors="pt"
        ).to(inputs_llm.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip().replace('<s> ','') for text in output_text]

        return output_text
    
# model = Blip2VicunaInstruct(dtype=torch.float32)
# print(get_parameter_number(model))
# from datasets.aokvqa_dataset import AOKVQADataset
# from torch.utils.data import Dataset, DataLoader
# dataset = AOKVQADataset()
# train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=16)
# optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr = 2e-5, betas=(0.9, 0.999), weight_decay=0.02)
# for step, data in enumerate(train_loader):
#     pixel_values = data['pixel_values']
#     samples = {
#             "text_input": data['questions'],
#             "text_output": data['open_answer_texts'],
#             "questions": data['questions'],
#             "qaprompts": data['questions'],
#             "pixel_values": pixel_values,
#         }
    
#     optimizer.zero_grad()
#     model.train()
#     outputs = model(samples)
#     loss = outputs['loss']
#     loss.backward()
#     optimizer.step()
#     print(loss)
#     pred_texts = model.generate(samples, max_new_tokens=50)
#     print(data['image_ids'])
#     print(data['questions'])
#     print(pred_texts)

#     if step==5:
#         break