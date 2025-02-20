import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # 可学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 可学习的平移参数
        self.eps = eps  # 防止除零的小常数

    def forward(self, x):
        # x: [..., normalized_shape]
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # [..., 1]
        variance = x.var(dim=-1, keepdim=True, unbiased=False)  # [..., 1]

        # 归一化
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)  # [..., normalized_shape]

        # 缩放和平移
        output = self.gamma * x_normalized + self.beta  # [..., normalized_shape]
        return output

if __name__ == '__main__':
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)

    # 定义参数
    batch_size = 1
    seq_length = 2
    d_model = 5
    normalized_shape = d_model  # 通常对最后一个维度进行归一化

    # 创建 MyLayerNorm 实例
    layer_norm = LayerNorm(normalized_shape)

    # 生成随机输入张量
    x = torch.randn(batch_size, seq_length, d_model)  # [batch_size, seq_length, d_model]

    # 计算层归一化输出
    output = layer_norm(x)

    # 打印输出张量的形状
    print("Output shape:", output.shape)  # 应该为: torch.Size([4, 10, 512]) 
    print(output)

    # 检查归一化后的均值和方差
    mean = output.mean(dim=-1)  # 计算最后一个维度的均值
    var = output.var(dim=-1, unbiased=False)  # 计算最后一个维度的方差
    
    print("\n归一化后的统计信息:")
    print("均值的平均值:", mean.mean().item(), "（理论上接近0）")
    print("均值的标准差:", mean.std().item(), "（理论上接近0）")
    print("方差的平均值:", var.mean().item(), "（理论上接近1）")
    print("方差的标准差:", var.std().item(), "（理论上接近0）")