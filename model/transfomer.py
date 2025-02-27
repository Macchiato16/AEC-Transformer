import torch
import torch.nn as nn
from model.encoder import Encoder
from model.positionalencoding import PositionalEncoding
import numpy as np
import librosa

class TransformerEchoCancellation(nn.Module):
    def __init__(self, 
                 input_dim=322,           # 输入特征维度 (2 * (n_fft // 2 + 1))
                 d_model=512,         # 模型维度
                 num_layers=6,        # 编码器层数
                 num_heads=8,         # 多头注意力头数
                 d_ff=1024,           # 前馈网络维度
                 dropout_prob=0.1,    # 丢弃率
                 max_len=50,        # 最大序列长度
                 freq_bins=161):      # 频率维度 (n_fft // 2 + 1)
        super(TransformerEchoCancellation, self).__init__()
        
        # 输入嵌入层，将输入特征映射到模型维度
        # 输入: [batch_size, seq_len, input_dim]
        # 输出: [batch_size, seq_len, d_model]
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码
        # 输入: [batch_size, seq_len, d_model]
        # 输出: [batch_size, seq_len, d_model]
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        
        # Transformer 编码器
        # 输入: [batch_size, seq_len, d_model]
        # 输出: [batch_size, seq_len, d_model]
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout_prob)
        
        # 输出层，生成理想掩码
        # 最终输出: [batch_size, seq_len, freq_bins]
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # [batch_size, seq_len, d_model//2]
            nn.ReLU(),
            nn.Linear(d_model // 2, freq_bins),  # [batch_size, seq_len, freq_bins]
            nn.Sigmoid()  # 掩码值应在 0-1 之间
        )
        
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入特征，形状为 [batch_size, seq_len, input_dim]
               包含拼接的远端信号特征和麦克风信号特征
            mask: 可选的掩码，用于注意力机制，形状为 [batch_size, 1, seq_len, seq_len]
            
        返回:
            掩码: 形状为 [batch_size, seq_len, freq_bins]，表示理想的掩码
        """
        # 嵌入层: [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model]
        x = self.embedding(x)
        
        # 添加位置编码: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)
        
        # 通过编码器: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        encoded = self.encoder(x, mask)
        
        # 生成掩码: [batch_size, seq_len, d_model] -> [batch_size, seq_len, freq_bins]
        mask_output = self.output_layer(encoded)
        
        return mask_output
    
    
