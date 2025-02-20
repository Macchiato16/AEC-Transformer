import torch
import torch.nn as nn
from model.attention import MultiHeadAttention  # 导入多头注意力
from model.feedforward import FeedForward  # 导入前馈网络
from model.layernorm import LayerNorm  # 导入层归一化

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_prob)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_prob)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        # 残差连接和层归一化
        x = self.norm1(x + attn_output)

        # 前馈网络
        ff_output = self.feed_forward(x)
        # 残差连接和层归一化
        x = self.norm2(x + ff_output)

        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_prob=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_prob) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

