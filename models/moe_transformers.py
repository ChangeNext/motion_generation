import torch
from typing import Any, List, Optional, Tuple, Union, Mapping
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from dataclasses import dataclass
import warnings


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of tensors. Shape: [batch_size, seqeunce_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None:
        return 0

    if isinstance(gate_logits, tuple):
        # cat along the layers?
        compute_device = gate_logits[0].device
        gate_logits = torch.cat([gate.to(compute_device) for gate in gate_logits], dim=0)

    routing_weights, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
    routing_weights = routing_weights.softmax(dim=-1)

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if selected_experts.dtype != torch.int64:
        selected_experts = selected_experts.to(torch.int64)

    if len(selected_experts.shape) == 2:
        selected_experts = selected_experts.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(routing_weights, axis=-1)
    
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert.unsqueeze(-1)) * (num_experts**2)

@dataclass
class MoeModelOutputWithPast:
    last_hidden_state: torch.FloatTensor = None
    # past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    noisy_logits: Optional[Tuple[torch.FloatTensor]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MoeCausalLMOutputWithPast:
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    noisy_logits: Optional[Tuple[torch.FloatTensor]] = None

class Expert(nn.Module):
    def __init__(self, embed_dim, fc_rate, drop_out_rate):
        super().__init__()
        
        self.w1 = nn.Linear(embed_dim, fc_rate*embed_dim)
        self.w2 = nn.Linear(fc_rate*embed_dim, embed_dim)
        self.w3 = nn.Linear(embed_dim, fc_rate*embed_dim)
        self.act_fn = nn.ReLU()
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, fc_rate* embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )   
    def forward(self, hidden_states):
        # current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        # current_hidden_states = self.dropout(self.w2(current_hidden_states))
        return self.net(hidden_states)

class NoisyTopkRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(embed_dim, num_experts)
        self.noise_linear = nn.Linear(embed_dim, num_experts)
        
    def forward(self, hidden_states):
        gate_logits = self.topkroute_linear(hidden_states)
        noise_logits = self.noise_linear(hidden_states)
        
        ##Adding scaled unit gaussian noise to the logtis
        noise = torch.randn_like(gate_logits)*F.softplus(noise_logits)
        noisy_logits = gate_logits + noise
        
        routing_weights = F.softmax(noisy_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        return routing_weights, selected_experts, noisy_logits, gate_logits
    
class MotionSparseMoeBlock(nn.Module):

    def __init__(self, embed_dim=512, drop_out_rate=0.1, fc_rate=4, num_experts=8, top_k=2):
        super().__init__()
        
        self.top_k = top_k
        self.embed_dim = embed_dim
        self.num_expert = num_experts
        
        self.router = NoisyTopkRouter(self.embed_dim, self.num_expert, self.top_k)
        self.experts = nn.ModuleList([Expert(self.embed_dim, fc_rate, drop_out_rate) for _ in range(num_experts)])
    
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.embed_dim)
        
        routing_weights, selected_experts, noisy_logits, gate_logits = self.router(hidden_states)
        # routing_weights: (batch * sequence_length, n_experts)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_expert).permute(2, 1, 0)
        
        # Reshape inputs for batch processing
        # Process each expert in parallel
        for expert_idx in range(self.num_expert):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, noisy_logits, gate_logits

class MotionDecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate =4, num_experts=8, top_k=2):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        self.input_layernorm = nn.LayerNorm(embed_dim)
        self.post_attention_layernorm = nn.LayerNorm(embed_dim)
        
        self.self_attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.block_sparse_moe = MotionSparseMoeBlock(embed_dim, drop_out_rate,fc_rate,num_experts, top_k)
    
    def forward(self, hidden_states):
        
        residual = hidden_states
        hidden_states = residual + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states, noisy_logits, router_logits = self.block_sparse_moe(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        outputs += (noisy_logits,)
        outputs += (router_logits,)
        return outputs
    
class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class MoEModelBase(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                num_experts=8,
                top_k=2):
        super().__init__()
        self.block_size = block_size
        self.drop = nn.Dropout(drop_out_rate)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        
        self.layers = nn.ModuleList([MotionDecoderLayer(embed_dim, block_size, n_head, drop_out_rate, fc_rate, num_experts, top_k) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        
        self.apply(self._init_weights)
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, clip_feature) -> Union[Tuple, MoeModelOutputWithPast]:
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            
        hidden_states = self.pos_embed(token_embeddings)
        all_router_logits = ()
        all_noise_router_logits = ()
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states)
            
            hidden_states = layer_outputs[0]
            all_noise_router_logits += (layer_outputs[1],)
            all_router_logits += (layer_outputs[-1],)

        return MoeModelOutputWithPast(
            last_hidden_state = hidden_states,
            noisy_logits=all_noise_router_logits,
            router_logits=all_router_logits
        )

class MoEModelHead(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                num_experts=8,
                top_k=2):
        super().__init__()
        
        self.block_size = block_size
        self.layers = nn.ModuleList([MotionDecoderLayer(embed_dim, block_size, n_head, drop_out_rate, fc_rate, num_experts, top_k) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        
        self.apply(self._init_weights)
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states) -> Union[Tuple, MoeModelOutputWithPast]:
        all_router_logits = hidden_states.router_logits
        all_noise_router_logits = hidden_states.noisy_logits
        hidden_states=hidden_states.last_hidden_state
        
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states)
            
            hidden_states = layer_outputs[0]
            all_noise_router_logits += (layer_outputs[1],)
            all_router_logits += (layer_outputs[-1],)

        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state = logits,
            noisy_logits=all_noise_router_logits,
            router_logits=all_router_logits
        )     
        
class Text2Motion_Transformer_MoE(nn.Module):
    def __init__(self, 
            num_vq=512, 
            embed_dim=1024, 
            clip_dim=512, 
            block_size=51, 
            num_layers=9, 
            n_head=16, 
            drop_out_rate=0.1, 
            fc_rate=4,
            num_experts=8, 
            top_k=2,
            router_aux_loss_coef = 0.001,
            router_z_loss_coef = 0.01):
        super().__init__()
        
        self.top_k = top_k
        self.num_vq = num_vq
        self.block_size = block_size
        self.num_experts = num_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.model = MoEModelBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, num_experts, top_k)
        
    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature):
        outputs = self.model(idxs, clip_feature)
        
        hidden_states = outputs.last_hidden_state
        
        aux_loss = load_balancing_loss_func(outputs.noisy_logits, self.num_experts, self.top_k)
        

        if isinstance(outputs.router_logits, tuple):
        # cat along the layers?
            compute_device = outputs.router_logits[0].device
            router_logits = torch.cat([gate.to(compute_device) for gate in outputs.router_logits], dim=0)
        router_z_loss = torch.logsumexp(router_logits, dim = -1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        
        loss = self.router_aux_loss_coef * aux_loss #+ self.router_z_loss_coef * router_z_loss
        
        logits = self.head(self.ln_f(hidden_states))
        return MoeCausalLMOutputWithPast(
            logits=logits,
            loss=loss,
        )
        
    def sample(self, clip_feature, if_categorial=False):
        for k in range(self.block_size):
            if k == 0:
                x = []
            else:
                x = xs
            outputs = self.forward(x, clip_feature)
            logits = outputs.logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)
            
            if k == self.block_size - 1:
                return xs[:, :-1]
        return xs

class Text2Motion_Cross_Transformer_MoE(nn.Module):
    def __init__(self, 
            num_vq=512, 
            embed_dim=1024, 
            clip_dim=512, 
            block_size=51, 
            num_layers=9, 
            n_head=16, 
            drop_out_rate=0.1, 
            fc_rate=4,
            num_experts=8, 
            top_k=2,
            router_aux_loss_coef = 0.001,
            router_z_loss_coef = 0.01):
        super().__init__()
        
        self.top_k = top_k
        self.num_vq = num_vq
        self.block_size = block_size
        self.num_experts = num_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.model_base = MoEModelBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, num_experts, top_k)
        self.model_head = MoEModelHead(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, num_experts, top_k)
        
        
    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature):
        feat = self.model_base(idxs, clip_feature)
        outputs = self.model_head(feat)
        
        aux_loss = load_balancing_loss_func(outputs.noisy_logits, self.num_experts, self.top_k)
        

        if isinstance(outputs.router_logits, tuple):
        # cat along the layers?
            compute_device = outputs.router_logits[0].device
            router_logits = torch.cat([gate.to(compute_device) for gate in outputs.router_logits], dim=0)
        router_z_loss = torch.logsumexp(router_logits, dim = -1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        
        loss = self.router_aux_loss_coef * aux_loss + self.router_z_loss_coef * router_z_loss
        logits = outputs.last_hidden_state
        return MoeCausalLMOutputWithPast(
            logits=logits,
            loss=loss,
        )
        
    def sample(self, clip_feature, if_categorial=False):
        for k in range(self.block_size):
            if k == 0:
                x = []
            else:
                x = xs
            outputs = self.forward(x, clip_feature)
            logits = outputs.logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)
            
            if k == self.block_size - 1:
                return xs[:, :-1]
        return xs

