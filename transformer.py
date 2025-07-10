import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.3):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.size()
        q = self.Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)


class Transformer_block(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.3):
        super().__init__()
        self.attention = attention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, input_dim, ff_dim, seq_len, action_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = positional_encoding(d_model, max_len=seq_len)
        self.value_head = nn.Linear(d_model, 1)

        self.encoder_block = nn.ModuleList([
            Transformer_block(d_model, n_heads, ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(d_model, action_dim)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))
        self.output = nn.Linear(d_model,1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        for block in self.encoder_block:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)

        # Return both policy logits and value
        mu = torch.tanh(self.mu_head(x))
        std = torch.sigmoid(self.log_std_param).expand_as(mu)
        value = self.value_head(x)
        return mu, std, value


class MultiScaleFusionTransformer(nn.Module):
    def __init__(self, encoder_15m, encoder_daily, d_model, n_classes=15):
        super().__init__()
        self.encoder_15m = encoder_15m
        self.encoder_daily = encoder_daily

        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, n_classes)  # logits for CrossEntropyLoss
        )

    def forward(self, x_15m, x_daily):
        z1 = self.encoder_15m(x_15m)   # [B, d_model]
        z2 = self.encoder_daily(x_daily)  # [B, d_model]
        fused = torch.cat([z1, z2], dim=-1)
        return self.fusion(fused)  # [B, n_classes]





















