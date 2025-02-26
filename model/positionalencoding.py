import os
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000,dropout=0.1):
        super(PositionalEncoding, self).__init__()
        # 使用公式: PE(pos,2i) = sin(pos/10000^(2i/d_model))
        # 使用公式: PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
        # 其中pos是位置,i是维度索引,d_model是embedding_dim
        self.dropout = nn.Dropout(p=dropout)
        # 初始化位置编码矩阵 shape: [max_len, embedding_dim]
        pe = torch.zeros(max_len, embedding_dim)
        # 生成位置索引 shape: [max_len, 1] 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码的除数项 shape: [embedding_dim/2]
        # 生成一个从0到embedding_dim-1,步长为2的等差数列,即[0,2,4,...],对应位置编码中的i
        # 将其乘以-ln(10000)/embedding_dim,然后取指数,得到10000^(-2i/d_model)
        # 这样可以得到位置编码公式中的分母项10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        # position * div_term shape: [max_len, embedding_dim/2]
        # 将sin函数应用于偶数索引位置 shape: [max_len, embedding_dim/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        # 将cos函数应用于奇数索引位置 shape: [max_len, embedding_dim/2] 
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加batch维度 shape: [1, max_len, embedding_dim]
        pe = pe.unsqueeze(0)
        # 注册位置编码矩阵为缓冲区
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        # self.pe[:, :x.size(1)] shape: [1, seq_len, embedding_dim]
        # 输出 shape: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1),:x.size(2)]
        return self.dropout(x)


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    # 创建位置编码实例
    plt.figure(figsize=(15, 8))
    # 创建位置编码实例
    embedding_dim = 20
    max_len = 5000
    pos_encoder = PositionalEncoding(embedding_dim, max_len,dropout=0.)

    # 获取位置编码矩阵
    pe = pos_encoder.pe.squeeze(0).numpy()  # 移除batch维度并转换为numpy数组

    # 创建图形
    
    # 选择几个不同维度的特征进行可视化
    plt.plot(np.arange(200),pe[0:200,4:8])
    plt.show()
