import torch
import torch.nn as nn
from model.encoder import Encoder
from model.positionalencoding import PositionalEncoding
import numpy as np
import librosa

class TransformerEchoCancellation(nn.Module):
    def __init__(self, 
                 input_dim,           # 输入特征维度 (2 * (n_fft // 2 + 1))
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
    
    def process_audio(self, far_end_features, mic_features):
        """
        处理音频特征并生成掩码
        
        参数:
            far_end_features: 远端信号特征，形状为 [batch_size, seq_len, freq_bins]
            mic_features: 麦克风信号特征，形状为 [batch_size, seq_len, freq_bins]
            
        返回:
            掩码: 用于消除回声的理想掩码，形状为 [batch_size, seq_len, freq_bins]
        """
        # 拼接远端信号特征和麦克风信号特征
        # [batch_size, seq_len, freq_bins] + [batch_size, seq_len, freq_bins] 
        # -> [batch_size, seq_len, 2*freq_bins]
        combined_features = torch.cat([far_end_features, mic_features], dim=-1)
        
        # 生成掩码: [batch_size, seq_len, 2*freq_bins] -> [batch_size, seq_len, freq_bins]
        mask = self(combined_features)
        
        return mask
    
    def apply_mask(self, mic_spectrogram, mask):
        """
        将掩码应用于麦克风信号频谱以消除回声
        
        参数:
            mic_spectrogram: 麦克风信号的频谱，形状为 [batch_size, seq_len, freq_bins]
            mask: 模型生成的掩码，形状为 [batch_size, seq_len, freq_bins]
            
        返回:
            enhanced_spectrogram: 增强后的信号频谱，形状为 [batch_size, seq_len, freq_bins]
        """
        # 应用掩码: [batch_size, seq_len, freq_bins] * [batch_size, seq_len, freq_bins]
        # -> [batch_size, seq_len, freq_bins]
        enhanced_spectrogram = mic_spectrogram * mask
        
        return enhanced_spectrogram
    
    def reconstruct_audio(self, enhanced_spectrogram, mic_phases, hop_length=256, win_length=512):
        """
        将增强后的频谱和原始相位重建为时域信号
        
        参数:
            enhanced_spectrogram: 增强后的频谱，形状为 [batch_size, seq_len, freq_bins]
            mic_phases: 麦克风信号的相位，形状为 [batch_size, seq_len, freq_bins]
            hop_length: STFT的跳跃长度
            win_length: STFT的窗口长度
            
        返回:
            reconstructed_audio: 重建的时域信号，形状为 [batch_size, n_samples]
        """
        batch_size = enhanced_spectrogram.shape[0]
        reconstructed_audio = []
        
        for i in range(batch_size):
            # 获取单个样本的频谱和相位
            spec = enhanced_spectrogram[i].cpu().numpy().T  # [freq_bins, seq_len]
            phase = mic_phases[i].cpu().numpy().T  # [freq_bins, seq_len]
            
            # 重建复数STFT
            complex_stft = spec * np.exp(1j * phase)  # [freq_bins, seq_len]
            
            # 逆STFT转换回时域
            audio = librosa.istft(complex_stft, hop_length=hop_length, 
                                 win_length=win_length, window='hann')  # [n_samples]
            
            reconstructed_audio.append(audio)
        
        # 将列表转换为张量
        reconstructed_audio = torch.tensor(np.array(reconstructed_audio), 
                                          device=enhanced_spectrogram.device)  # [batch_size, n_samples]
        
        return reconstructed_audio
    
    def train_step(self, far_end_features, mic_features, target_masks, optimizer, criterion):
        """
        执行单个训练步骤
        
        参数:
            far_end_features: 远端信号特征，形状为 [batch_size, seq_len, freq_bins]
            mic_features: 麦克风信号特征，形状为 [batch_size, seq_len, freq_bins]
            target_masks: 目标掩码，形状为 [batch_size, seq_len, freq_bins]
            optimizer: 优化器
            criterion: 损失函数
            
        返回:
            loss: 当前批次的损失值
        """
        # 将特征移动到模型所在的设备
        device = next(self.parameters()).device
        far_end_features = far_end_features.to(device)
        mic_features = mic_features.to(device)
        target_masks = target_masks.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        # [batch_size, seq_len, freq_bins] + [batch_size, seq_len, freq_bins] 
        # -> [batch_size, seq_len, 2*freq_bins]
        combined_features = torch.cat([far_end_features, mic_features], dim=-1)
        
        # [batch_size, seq_len, 2*freq_bins] -> [batch_size, seq_len, freq_bins]
        predicted_masks = self(combined_features)
        
        # 计算损失
        loss = criterion(predicted_masks, target_masks)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        return loss.item()
    
    def validate(self, far_end_features, mic_features, target_masks, criterion):
        """
        在验证集上评估模型
        
        参数:
            far_end_features: 远端信号特征，形状为 [batch_size, seq_len, freq_bins]
            mic_features: 麦克风信号特征，形状为 [batch_size, seq_len, freq_bins]
            target_masks: 目标掩码，形状为 [batch_size, seq_len, freq_bins]
            criterion: 损失函数
            
        返回:
            loss: 验证损失
        """
        # 将特征移动到模型所在的设备
        device = next(self.parameters()).device
        far_end_features = far_end_features.to(device)
        mic_features = mic_features.to(device)
        target_masks = target_masks.to(device)
        
        # 设置为评估模式
        self.eval()
        
        with torch.no_grad():
            # 前向传播
            # [batch_size, seq_len, freq_bins] + [batch_size, seq_len, freq_bins] 
            # -> [batch_size, seq_len, 2*freq_bins]
            combined_features = torch.cat([far_end_features, mic_features], dim=-1)
            
            # [batch_size, seq_len, 2*freq_bins] -> [batch_size, seq_len, freq_bins]
            predicted_masks = self(combined_features)
            
            # 计算损失
            loss = criterion(predicted_masks, target_masks)
        
        # 恢复训练模式
        self.train()
        
        return loss.item()
    
    def inference(self, far_end_features, mic_features, mic_phases):
        """
        执行推理并重建增强的音频
        
        参数:
            far_end_features: 远端信号特征，形状为 [batch_size, seq_len, freq_bins]
            mic_features: 麦克风信号特征，形状为 [batch_size, seq_len, freq_bins]
            mic_phases: 麦克风信号的相位，形状为 [batch_size, seq_len, freq_bins]
            
        返回:
            enhanced_spectrogram: 增强后的频谱，形状为 [batch_size, seq_len, freq_bins]
        """
        # 将特征移动到模型所在的设备
        device = next(self.parameters()).device
        far_end_features = far_end_features.to(device)
        mic_features = mic_features.to(device)
        
        # 设置为评估模式
        self.eval()
        
        with torch.no_grad():
            # 前向传播
            # [batch_size, seq_len, freq_bins] + [batch_size, seq_len, freq_bins] 
            # -> [batch_size, seq_len, 2*freq_bins]
            combined_features = torch.cat([far_end_features, mic_features], dim=-1)
            
            # [batch_size, seq_len, 2*freq_bins] -> [batch_size, seq_len, freq_bins]
            predicted_masks = self(combined_features)
            
            # 应用掩码到麦克风频谱
            # [batch_size, seq_len, freq_bins] * [batch_size, seq_len, freq_bins]
            # -> [batch_size, seq_len, freq_bins]
            enhanced_spectrogram = mic_features * predicted_masks
        
        # 恢复训练模式
        self.train()
        
        return enhanced_spectrogram
