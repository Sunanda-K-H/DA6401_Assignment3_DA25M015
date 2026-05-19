"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_DRIVE_ID = "1Uv2X_qFSnUxX1BYw5N7TzE_fCd8Ek8fz"


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    # raise NotImplementedError
    d_k = Q.shape[-1]  # key dimension
    dot_prod = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)  # scaled scores
    if mask is not None:
        dot_prod = dot_prod.masked_fill(mask, float('-inf'))  # hide masked tokens
    attn_w = torch.softmax(dot_prod, dim=-1) # (..., seq_q, seq_k)
    output = torch.matmul(attn_w, V) # (..., seq_q, d_v)

    return (output, attn_w)

# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # raise NotImplementedError
    batch,src_len = src.shape
    mask = (src == pad_idx)  # source padding positions
    mask = mask.reshape(batch,1,1,src_len)  # broadcast for attention
    return mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    # raise NotImplementedError
    batch,tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx)  # target padding positions
    pad_mask = pad_mask.reshape(batch,1,1,tgt_len)  # broadcast padding mask
    causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device), diagonal=1)  # future tokens
    causal_mask = causal_mask.reshape(1,1,tgt_len,tgt_len)  # broadcast causal mask
    mask = pad_mask | causal_mask  # combined decoder mask
    return mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, scale_attention: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.scale_attention = scale_attention
        # raise NotImplementedError

        self.wq = nn.Linear(self.d_model,self.d_model)  # query projection
        self.wk = nn.Linear(self.d_model,self.d_model)  # key projection
        self.wv = nn.Linear(self.d_model,self.d_model)  # value projection
        self.wo = nn.Linear(self.d_model,self.d_model)  # output projection
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        # raise NotImplementedError

        batch,seq_q, d_model = query.shape  # query length
        _,seq_k,_ = key.shape  # key length

        q = self.wq(query)  # projected queries
        k = self.wk(key)  # projected keys
        v = self.wv(value)  # projected values

        q = q.reshape(batch, seq_q, self.num_heads, self.d_k)  # split heads
        k = k.reshape(batch, seq_k, self.num_heads, self.d_k)  # split heads
        v = v.reshape(batch, seq_k, self.num_heads, self.d_k)  # split heads

        q = q.permute(0,2,1,3)  # heads before sequence
        k = k.permute(0,2,1,3)  # heads before sequence
        v = v.permute(0,2,1,3)  # heads before sequence


        atten = torch.matmul(q, k.transpose(-2,-1)) # batch, num_heads, seq_q, seq_k
        if self.scale_attention:
            atten = atten / math.sqrt(self.d_k)  # scale logits
        if mask is not None:
            inf_mask = torch.where(mask, float('-inf'), 0.0)  # mask values
            atten += inf_mask  # apply mask
        sf_atten = F.softmax(atten, dim=-1)  # attention weights
        self.attn_weights = sf_atten.detach()  # save for heatmaps
        sf_atten = self.dropout(sf_atten)  # attention dropout
        final_atten = torch.matmul(sf_atten, v) # batch, num_heads, seq_q, d_k
        final_atten = final_atten.permute(0,2,1,3)  # restore sequence order
        final_atten = final_atten.reshape(batch, seq_q, d_model)  # merge heads
        out = self.wo(final_atten)  # final projection
        return out


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════


# https://towardsdev.com/positional-encoding-in-transformers-using-pytorch-63b5c3f57d54

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        # raise NotImplementedError

        pe = torch.zeros(max_len, d_model)  # positional table
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # frequencies
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        pe = pe.unsqueeze(0)  # batch dimension
        self.register_buffer('pe', pe)  # non-trainable buffer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        # raise NotImplementedError
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        #   self.linear1 = nn.Linear(d_model, d_ff)
        #   self.linear2 = nn.Linear(d_ff, d_model)
        #   self.dropout = nn.Dropout(p=dropout)
        # raise NotImplementedError
        self.w1 = nn.Linear(d_model, d_ff)  # expand dimension
        self.w2 = nn.Linear(d_ff, d_model)  # project back
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        # raise NotImplementedError
        o1 = self.w1(x)  # first linear layer
        h1 = F.relu(o1)  # ReLU activation
        h1 = self.dropout(h1)  # dropout between layers
        o2 = self.w2(h1)  # second linear layer
        return o2


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, scale_attention: bool = True) -> None:
        super().__init__()
        # TODO:instantiate:
        # raise NotImplementedError
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.self_attention = MultiHeadAttention(self.d_model, self.num_heads, dropout, scale_attention)
        self.ffn = PositionwiseFeedForward(self.d_model, self.d_ff, dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # raise NotImplementedError
        matten = self.self_attention(x,x,x,src_mask)  # encoder self-attention
        matten = self.dropout(matten)  # attention dropout
        ln = self.ln1(x + matten)  # residual add and norm
        ff = self.ffn(ln)  # feed-forward block
        ff = self.dropout(ff)  # FFN dropout
        out = self.ln2(ln+ff)  # residual add and norm
        return out


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, scale_attention: bool = True) -> None:
        super().__init__()
        # TODO: instantiate:
        # raise NotImplementedError
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, scale_attention)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout, scale_attention)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # raise NotImplementedError

        self_attn = self.self_attention(x, x, x, tgt_mask)  # masked self-attention
        x = self.ln1(x + self.dropout(self_attn))  # residual add and norm

        cross_attn = self.cross_attention(x, memory, memory, src_mask)  # encoder-decoder attention
        x = self.ln2(x + self.dropout(cross_attn))  # residual add and norm

        ffn_out = self.ffn(x)  # feed-forward block
        x = self.ln3(x + self.dropout(ffn_out))  # residual add and norm

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        # raise NotImplementedError
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])  # encoder stack
        self.norm = nn.LayerNorm(layer.d_model)  # final norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        # raise NotImplementedError
        for layer in self.layers:
            x = layer(x, mask)  # one encoder layer
            x = self.norm(x)  # normalize layer output

        return x

class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        # raise NotImplementedError
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])  # decoder stack

        self.norm = nn.LayerNorm(layer.d_model)  # final norm

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        # raise NotImplementedError
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)  # one decoder layer

        out = self.norm(x)  # normalize decoder output

        return out


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = None,
        scale_attention: bool = True,
        pos_encoding_type: str = "sinusoidal",
    ) -> None:
        super().__init__()
        # TODO: Instantiate
        # raise NotImplementedError
        load_path = checkpoint_path
        if src_vocab_size is None or tgt_vocab_size is None:
            default_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
            load_path = default_path if os.path.exists(default_path) else "checkpoint.pt"
            if not os.path.exists(load_path):
                gdown.download(id=CHECKPOINT_DRIVE_ID, output=load_path, quiet=False)
            checkpoint = torch.load(load_path, map_location="cpu")
            config = checkpoint["model_config"]
            src_vocab_size = config["src_vocab_size"]
            tgt_vocab_size = config["tgt_vocab_size"]
            d_model = config.get("d_model", d_model)
            N = config.get("N", N)
            num_heads = config.get("num_heads", num_heads)
            d_ff = config.get("d_ff", d_ff)
            dropout = config.get("dropout", dropout)
            scale_attention = config.get("scale_attention", scale_attention)
            pos_encoding_type = config.get("pos_encoding_type", pos_encoding_type)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.scale_attention = scale_attention
        self.pos_encoding_type = pos_encoding_type

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.learned_pos_embed = nn.Embedding(5000, d_model) if pos_encoding_type == "learned" else None

        encoder_layer = EncoderLayer(d_model,num_heads,d_ff,dropout,scale_attention)

        decoder_layer = DecoderLayer(d_model,num_heads,d_ff,dropout,scale_attention)

        self.encoder = Encoder(encoder_layer, N)

        self.decoder = Decoder(decoder_layer, N)

        self.proj = nn.Linear(d_model, tgt_vocab_size)

        if load_path is not None:
            if not os.path.exists(load_path):
                gdown.download(id=CHECKPOINT_DRIVE_ID, output=load_path, quiet=False)
            checkpoint = torch.load(load_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.load_state_dict(state_dict)

    def _load_infer_assets(self):
        if hasattr(self, "src_vocab") and hasattr(self, "tgt_vocab"):
            return
        import spacy
        from dataset import Multi30kDataset
        data = Multi30kDataset(split="train")
        data.build_vocab()
        self.src_vocab = data.src_vocab
        self.tgt_vocab = data.tgt_vocab
        self.tgt_itos = data.tgt_itos
        self.src_tokenizer = data.spacy_de

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        # raise NotImplementedError
        src_embed = self.src_embed(src)

        src_embed = src_embed * math.sqrt(self.d_model)

        if self.learned_pos_embed is not None:
            positions = torch.arange(0, src_embed.size(1), device=src_embed.device).unsqueeze(0)
            src_embed = src_embed + self.learned_pos_embed(positions)
            src_embed = self.pos_encoding.dropout(src_embed)
        else:
            src_embed = self.pos_encoding(src_embed)

        memory = self.encoder(src_embed, src_mask)

        return memory

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # raise NotImplementedError
        tgt_embed = self.tgt_embed(tgt)

        tgt_embed = tgt_embed * math.sqrt(self.d_model)

        if self.learned_pos_embed is not None:
            positions = torch.arange(0, tgt_embed.size(1), device=tgt_embed.device).unsqueeze(0)
            tgt_embed = tgt_embed + self.learned_pos_embed(positions)
            tgt_embed = self.pos_encoding.dropout(tgt_embed)
        else:
            tgt_embed = self.pos_encoding(tgt_embed)

        decoder_out = self.decoder(tgt_embed,memory,src_mask,tgt_mask)

        logits = self.proj(decoder_out)

        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # raise NotImplementedError
        memory = self.encode(src, src_mask)

        logits = self.decode(memory,src_mask,tgt,tgt_mask)

        return logits


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        # raise NotImplementedError
        self.eval()
        self._load_infer_assets()

        device = next(self.parameters()).device

        bos_idx = self.tgt_vocab["<sos>"]
        eos_idx = self.tgt_vocab["<eos>"]
        pad_idx = self.src_vocab["<pad>"]

        src_tokens = [tok.text.lower() for tok in self.src_tokenizer.tokenizer(src_sentence)]

        src_ids = [self.src_vocab["<sos>"]]

        for tok in src_tokens:
            src_ids.append(self.src_vocab.get(tok, self.src_vocab["<unk>"]))

        src_ids.append(self.src_vocab["<eos>"])

        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

        src_mask = make_src_mask(src, pad_idx).to(device)

        with torch.no_grad():

            memory = self.encode(src, src_mask)

            ys = torch.ones(1, 1, dtype=torch.long, device=device)
            ys = ys * bos_idx

            for i in range(100):

                tgt_mask = make_tgt_mask(ys, self.tgt_vocab["<pad>"]).to(device)

                logits = self.decode(memory, src_mask, ys, tgt_mask)

                next_logits = logits[:, -1, :]

                next_word = torch.argmax(next_logits, dim=-1)

                next_word = next_word.unsqueeze(1)

                ys = torch.cat([ys, next_word], dim=1)

                if next_word.item() == eos_idx:
                    break

        out_ids = ys.squeeze(0).tolist()

        out_tokens = []

        for idx in out_ids:

            if idx == bos_idx or idx == self.tgt_vocab["<pad>"]:
                continue

            if idx == eos_idx:
                break

            if hasattr(self.tgt_vocab, "lookup_token"):
                out_tokens.append(self.tgt_vocab.lookup_token(idx))
            elif hasattr(self.tgt_vocab, "itos"):
                out_tokens.append(self.tgt_vocab.itos[idx])
            else:
                out_tokens.append(self.tgt_itos[idx])

        return " ".join(out_tokens)
