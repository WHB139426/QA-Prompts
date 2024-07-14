import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
import os
from transformers import BertModel, RobertaModel, AutoConfig, AutoTokenizer, DebertaV2Model
from transformers.models.perceiver.modeling_perceiver import PerceiverLayer
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *

class SelfAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.layer_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape (b, n, d)
        """
        residual_x  = x
        x = self.layer_norm(x) # [b, n1, D]
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out) + residual_x

        return out

class QueryCrossAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.layer_norm_query = nn.LayerNorm(dim)
        self.layer_norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, query, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query, kv (torch.Tensor): shape (b, n, d)
        """
        kv = self.layer_norm_kv(kv)
        residual_query  = query
        query = self.layer_norm_query(query) # [b, n, D]
        h = self.heads
        q = self.to_q(query)
        kv_input = torch.cat((kv, query), dim=-2)

        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out) + residual_query

        return out
    
class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim)
        self.dense1 = nn.Linear(dim, dim*mult, bias=True)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(dim*mult, dim, bias=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape (b, n, D)
        """
        residual_x  = x
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        
        return x + residual_x
    
class SamplerBlock(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int, mult: int = 4):
        super().__init__()
        self.self_attn = SelfAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.cross_attn = QueryCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.ffn = FFN(dim=dim, mult=mult)

    def forward(self, query_tokens, kv_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query, kv (torch.Tensor): shape (b, n, d)
        """
        query_tokens = self.self_attn(query_tokens)
        query_tokens = self.cross_attn(query_tokens, kv_input)
        query_tokens = self.ffn(query_tokens)

        return query_tokens

class SamplerFormer(nn.Module):
    def __init__(
            self, 
            depth: int = 2, 
            dim: int = 768, 
            dim_head: int = int(768/12), 
            heads: int = 12, 
            mult: int = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                SamplerBlock(
                    dim=dim, dim_head=dim_head, heads=heads, mult=mult
                )
            )
        self.norm = nn.LayerNorm(dim)
        self.latents = nn.Parameter(torch.load("./experiments/query_tokens_vicuna.pth", map_location='cpu')[0]) # [num_latents, 768]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n, D)
        Returns:
            shape (b, n, D) where n is self.num_latents
        """
        b, n, d = x.shape
        # blocks
        latents = repeat(self.latents, "n d -> b n d", b=b)
        for block in self.layers:
            latents = block(query_tokens=latents, kv_input=x)
        return self.norm(latents) 
    
class QAPrompting(nn.Module):
    def __init__(self, image_dim=1408):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=False)
        self.text_encoder = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')

        self.image_dense = nn.Linear(image_dim, 768)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=768, kdim=768, vdim=768, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*768, 768)
        self.sigmoid = nn.Sigmoid()

        self.decoder = SamplerFormer(depth=2, dim=768, dim_head= int(768/4), heads=4, mult=4)
        self.llm_proj = nn.Linear(768, 4096)
        self.llm_proj.load_state_dict(torch.load("./experiments/llm_proj_vicuna.pth", map_location='cpu'))

        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        self.text_encoder.eval()

    def encode_text(self, texts):
        text_inputs = self.tokenizer(texts, padding='longest', truncation=True, max_length=512, return_tensors="pt").to(self.device)
        text_outputs = self.text_encoder(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
        word_text_embeds = text_outputs.last_hidden_state # [bs, seq_len, text_hidden_size]
        return word_text_embeds

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def forward(self, image_embeds, inputs_txt):
        hidden_states = self.encode_text(inputs_txt) # [bs, seq_len, 768]

        image_embedding = self.image_dense(image_embeds) # [bs, 257, 768]

        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding) # [bs, seq_len, 768]
        merge = torch.cat([hidden_states, image_att], dim=-1) # [bs, seq_len, 768*2]
        gate = self.sigmoid(self.gate_dense(merge)) # [bs, seq_len, 768]
        hidden_states = (1 - gate) * hidden_states + gate * image_att # [bs, seq_len, 768]

        query_tokens = self.decoder(hidden_states) # [bs, 32, 768]
        query_tokens = self.llm_proj(query_tokens) # [bs, 32, 4096]
        return query_tokens
    
# model = QAPrompting(1024)
# image_embeds = torch.randn(2, 576, 1024)
# inputs_txt = ['Question: What is the man wearing on his shoulders? Short answer: jacket', 'Question: What type of clothing is this man wearing? Short answer: wetsuit']
# query_tokens = model(image_embeds, inputs_txt)
# print(query_tokens.shape)