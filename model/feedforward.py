import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_prob=0.1):
        super(FeedForward, self).__init__()
        # 定义第一个线性层，将输入维度从 d_model 映射到 d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        # 定义第二个线性层，将维度从 d_ff 映射回 d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        # 定义 ReLU 激活函数
        self.relu = nn.ReLU()
        # 定义 Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        # 通过第一个线性层和 ReLU 激活函数
        x = self.relu(self.linear1(x))  # [batch_size, seq_length, d_ff]
        # 应用 Dropout
        x = self.dropout(x)
        # 通过第二个线性层
        x = self.linear2(x)  # [batch_size, seq_length, d_model]
        return x
