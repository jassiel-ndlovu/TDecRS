"""SASRec, BERT4Rec and the DCR adaptor.

Both backbones are compact re-implementations, written to follow the papers'
architectures and ML-1M defaults:
  SASRec   -- Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
              Block layout follows the authors' PyTorch port (pmixer/SASRec.pytorch,
              Apache-2.0): pre-LN attention, point-wise FFN with residual,
              final LayerNorm, shared item embedding for scoring.
  BERT4Rec -- Sun et al., "BERT4Rec", CIKM 2019. Post-LN Transformer encoder with
              GELU, learned positions, cloze masking, and an output head
              (dense + GELU + LayerNorm) scored against the tied item embedding
              plus a per-item bias, as in the authors' TF code.
No third-party code is vendored.

Interface shared by both backbones:
  encode(input_ids)            -> H (B, L, d), the representation before scoring
  logits(h, item_ids=None)     -> scores against all items (or the given ones)
The adaptor sits between the two.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCRAdaptor(nn.Module):
    """Mediator injection before the scoring layer.

    DecFM (code/ml-1m/model.py) builds M(d, u) as the FM bi-interaction over the
    user feature embeddings and the confounder embeddings scaled by d_g, then
    adds M as one extra feature of the outer FM, which contributes <M, e_i> to
    the score. Here the user is represented by a single vector h (the sequence
    representation at the prediction position), so with v = {h, d_1 e_1, ...,
    d_G e_G}:

        M = sum_{j<l} v_j * v_l = h * S + 0.5 * (S^2 - Q),
        S = sum_g d_g e_g,   Q = sum_g d_g^2 e_g^2

    and the score becomes <h + M, e_i>. The only new parameters are the G x d
    confounder embedding, as in DecFM.

    mode 'none'   -> identity, no parameters
    mode 'static' -> d = dbar (the trainer passes the prior broadcast per position)
    mode 'dcr'    -> d = d_u^(tau) per position
    'static' and 'dcr' run the same code on different d, so DCR with kappa=inf
    reproduces 'static' bit for bit (checked in smoke_test.py).
    """

    def __init__(self, mode, num_groups, dim, init_std=0.02):
        super().__init__()
        assert mode in ("none", "static", "dcr")
        self.mode = mode
        if mode != "none":
            self.confounder_embed = nn.Embedding(num_groups, dim)
            nn.init.normal_(self.confounder_embed.weight, std=init_std)

    def mediator(self, h, conf):
        E = self.confounder_embed.weight             # (G, d)
        S = conf @ E                                 # (..., d)
        Q = (conf * conf) @ (E * E)
        return h * S + 0.5 * (S * S - Q)

    def forward(self, h, conf=None):
        if self.mode == "none":
            return h
        return h + self.mediator(h, conf)


class PointWiseFF(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d, d), nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.l2(self.drop(F.relu(self.l1(x)))))


class SASRec(nn.Module):
    causal = True

    def __init__(self, num_items, max_len=200, dim=50, num_blocks=2, num_heads=1, dropout=0.2):
        super().__init__()
        self.num_items, self.dim = num_items, dim
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.emb_drop = nn.Dropout(dropout)
        self.attn_ln = nn.ModuleList([nn.LayerNorm(dim, eps=1e-8) for _ in range(num_blocks)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
                                   for _ in range(num_blocks)])
        self.ffn_ln = nn.ModuleList([nn.LayerNorm(dim, eps=1e-8) for _ in range(num_blocks)])
        self.ffn = nn.ModuleList([PointWiseFF(dim, dropout) for _ in range(num_blocks)])
        self.last_ln = nn.LayerNorm(dim, eps=1e-8)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, ids):
        L = ids.size(1)
        x = self.item_emb(ids) * math.sqrt(self.dim)
        x = x + self.pos_emb.weight[:L].unsqueeze(0)
        keep = (ids > 0).unsqueeze(-1).to(x.dtype)
        x = self.emb_drop(x) * keep
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ids.device), 1)
        for ln_a, attn, ln_f, ffn in zip(self.attn_ln, self.attn, self.ffn_ln, self.ffn):
            q = ln_a(x)
            a, _ = attn(q, x, x, attn_mask=causal, need_weights=False)
            x = ffn(ln_f(q + a)) * keep
        return self.last_ln(x)

    def logits(self, h, item_ids=None):
        if item_ids is None:
            return h @ self.item_emb.weight.t()                   # (..., N+1)
        return (h.unsqueeze(-2) * self.item_emb(item_ids)).sum(-1)


class BERT4Rec(nn.Module):
    causal = False

    def __init__(self, num_items, max_len=200, dim=64, num_blocks=2, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_items, self.dim = num_items, dim
        self.mask_id = num_items + 1
        self.item_emb = nn.Embedding(num_items + 2, dim, padding_idx=0)   # + [MASK]
        self.pos_emb = nn.Embedding(max_len, dim)
        self.emb_ln = nn.LayerNorm(dim, eps=1e-12)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, num_heads, 4 * dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=False, layer_norm_eps=1e-12)
        self.encoder = nn.TransformerEncoder(layer, num_blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim, eps=1e-12))
        self.out_bias = nn.Parameter(torch.zeros(num_items + 1))
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.trunc_normal_(p, std=0.02, a=-0.04, b=0.04)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, ids):
        L = ids.size(1)
        x = self.item_emb(ids) + self.pos_emb.weight[:L].unsqueeze(0)
        x = self.emb_drop(self.emb_ln(x))
        x = self.encoder(x, src_key_padding_mask=(ids == 0))
        return self.head(x)

    def logits(self, h, item_ids=None):
        E = self.item_emb.weight[:self.num_items + 1]           # no score for [MASK]
        if item_ids is None:
            return h @ E.t() + self.out_bias
        return (h.unsqueeze(-2) * E[item_ids]).sum(-1) + self.out_bias[item_ids]


class SeqRec(nn.Module):
    """backbone + adaptor; the adaptor acts on encode()'s output."""

    def __init__(self, backbone, adaptor):
        super().__init__()
        self.backbone, self.adaptor = backbone, adaptor

    def represent(self, ids, conf=None):
        return self.adaptor(self.backbone.encode(ids), conf)


def build(name, num_items, num_groups, adaptor, max_len, dim=None, num_blocks=2,
          num_heads=None, dropout=None):
    if name == "sasrec":
        bb = SASRec(num_items, max_len, dim or 50, num_blocks, num_heads or 1,
                    0.2 if dropout is None else dropout)
    elif name == "bert4rec":
        bb = BERT4Rec(num_items, max_len, dim or 64, num_blocks, num_heads or 2,
                      0.1 if dropout is None else dropout)
    else:
        raise ValueError(name)
    return SeqRec(bb, DCRAdaptor(adaptor, num_groups, bb.dim))
