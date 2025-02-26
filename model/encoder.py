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
        # Pre-LayerNorm 结构
        # 先进行层归一化，然后是多头自注意力
        attn_input = self.norm1(x)
        attn_output = self.self_attn(attn_input, attn_input, attn_input, mask)
        # 残差连接(attn_output已经经过dropout)
        x = x + attn_output

        # 先进行层归一化，然后是前馈网络
        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        # 残差连接(ff_output已经经过dropout)
        x = x + ff_output

        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_prob=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_prob) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)  # 最终的层归一化

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        # 最后进行一次层归一化
        return self.norm(x)

