import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须是 num_heads 的整数倍"

        self.d_model = d_model  # 模型的维度
        self.num_heads = num_heads  # 注意力头的数量
        self.d_k = d_model // num_heads  # 每个注意力头的维度

        # 定义线性变换层，用于将输入映射到 Q, K, V
        self.W_q = nn.Linear(d_model, d_model)  # Q 线性变换
        self.W_k = nn.Linear(d_model, d_model)  # K 线性变换
        self.W_v = nn.Linear(d_model, d_model)  # V 线性变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出线性变换

        self.dropout = nn.Dropout(dropout_prob) # dropout

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, num_heads, seq_length, d_k]
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # attn_scores: [batch_size, num_heads, seq_length, seq_length]

        # 应用掩码（可选）
        if mask is not None:
            # mask: [batch_size, 1, seq_length, seq_length] 
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # attn_probs: [batch_size, num_heads, seq_length, seq_length]

        # 计算加权后的值
        output = torch.matmul(attn_probs, V)
        # output: [batch_size, num_heads, seq_length, d_k]
        return self.dropout(output) # 应用 dropout

    def split_heads(self, x):
        # x: [batch_size, seq_length, d_model]
        batch_size, seq_length, d_model = x.size()
        # 将 d_model 维度拆分为 num_heads * d_k
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        # 返回: [batch_size, num_heads, seq_length, d_k]

    def combine_heads(self, x):
        # x: [batch_size, num_heads, seq_length, d_k]
        batch_size, _, seq_length, d_k = x.size()
        # 将 num_heads 和 d_k 维度合并回 d_model
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        # 返回: [batch_size, seq_length, d_model]

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, seq_length, d_model]

        # 线性变换并拆分多头
        Q = self.split_heads(self.W_q(Q))  # [batch_size, num_heads, seq_length, d_k]
        K = self.split_heads(self.W_k(K))  # [batch_size, num_heads, seq_length, d_k]
        V = self.split_heads(self.W_v(V))  # [batch_size, num_heads, seq_length, d_k]

        # 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: [batch_size, num_heads, seq_length, d_k]

        # 合并多头并进行线性变换
        output = self.W_o(self.combine_heads(attn_output))
        # output: [batch_size, seq_length, d_model]
        return self.dropout(output) # 应用 dropout

if __name__ == '__main__':
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)

    # 定义模型参数
    d_model = 512  # 模型的维度
    num_heads = 8  # 注意力头的数量
    batch_size = 4  # 批次大小
    seq_length = 10  # 序列长度
    dropout_prob = 0.2 # dropout 概率

    # 创建 MultiHeadAttention 实例
    attention = MultiHeadAttention(d_model, num_heads, dropout_prob)

    # 生成随机输入张量
    Q = torch.randn(batch_size, seq_length, d_model)  # [batch_size, seq_length, d_model]
    K = torch.randn(batch_size, seq_length, d_model)  # [batch_size, seq_length, d_model]
    V = torch.randn(batch_size, seq_length, d_model)  # [batch_size, seq_length, d_model]

    # 生成随机掩码（可选）
    # 这里生成一个掩码，有 50% 的概率为 0（表示被屏蔽）
    mask = (torch.rand(batch_size, 1, 1, seq_length) > 0.5).int() # [batch_size, 1, 1, seq_length]

    # 计算多头注意力
    output = attention(Q, K, V, mask)

    # 打印输出张量的形状
    print("Output shape:", output.shape)  # 应该为: torch.Size([4, 10, 512])





