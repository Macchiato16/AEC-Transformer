import os
import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
import h5py
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入模型和预处理类
from preprocess import Config, load_and_resample, compute_stft, compute_log_power_spec
from model.transfomer import TransformerEchoCancellation


def parse_args():
    parser = argparse.ArgumentParser(description='回声消除模型推理测试')
    parser.add_argument('--farend_path', type=str, default='f00000_farend.wav', help='远端信号文件路径')
    parser.add_argument('--mic_path', type=str, default='output1.wav', help='麦克风信号文件路径')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_final.pt', help='模型权重文件路径')
    parser.add_argument('--output_path', type=str, default='output.wav', help='输出文件路径')
    parser.add_argument('--use_cuda', action='store_true', help='使用CUDA')
    parser.add_argument('--overlap', action='store_true', help='是否使用重叠帧进行增强效果')
    parser.add_argument('--griffin_lim', action='store_true', help='使用Griffin-Lim算法重建相位')
    parser.add_argument('--n_iter', type=int, default=50, help='Griffin-Lim算法的迭代次数')
    return parser.parse_args()

def create_model(config):
    # 创建模型实例
    model = TransformerEchoCancellation(
        input_dim=config.feature_dim * 2,  # 远端信号和麦克风信号的特征拼接
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        dropout_prob=0.1,
        max_len=config.max_seq_len,
        freq_bins=config.feature_dim
    )
    return model

def load_model(model_path, config, device):
    # 创建模型
    model = create_model(config)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 兼容不同保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"模型已加载: {model_path}")
    return model

def preprocess_for_inference(farend_audio, mic_audio, config):
    """
    预处理音频信号，生成模型输入特征
    """
    # 计算STFT
    farend_mag, farend_phase = compute_stft(
        farend_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    mic_mag, mic_phase = compute_stft(
        mic_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    
    # 计算对数功率谱
    farend_log_power = compute_log_power_spec(farend_mag, config.min_db, config.ref_db)
    mic_log_power = compute_log_power_spec(mic_mag, config.min_db, config.ref_db)
    
    # 拼接特征（麦克风信号和远端信号的对数功率谱）
    features = np.concatenate([mic_log_power.T, farend_log_power.T], axis=1)
    
    return features, mic_mag, mic_phase

def split_into_sequences(features, max_seq_len, overlap=False):
    """
    将特征拆分为多个固定长度的序列
    
    Args:
        features: 特征矩阵，形状为 [T, F*2]
        max_seq_len: 每个序列的最大长度
        overlap: 是否使用重叠拆分（用于推理阶段提高质量）
    
    Returns:
        sequences: 拆分后的序列列表
        seq_lengths: 每个序列的有效长度
        stride: 序列重建时的步长
    """
    T, F = features.shape
    
    # 确定步长
    stride = max_seq_len // 2 if overlap else max_seq_len
    
    # 计算需要多少个序列
    n_sequences = int(np.ceil(T / stride))
    
    # 创建空序列数组
    padded_length = (n_sequences - 1) * stride + max_seq_len
    padded_features = np.zeros((padded_length, F))
    padded_features[:T] = features
    
    # 拆分为多个序列
    sequences = []
    seq_positions = []  # 记录每个序列的原始位置
    
    for i in range(0, padded_length - max_seq_len + 1, stride):
        seq = padded_features[i:i+max_seq_len]
        sequences.append(seq)
        
        # 记录该序列在原始信号中的起始位置和有效长度
        valid_length = min(max_seq_len, T - i) if i < T else 0
        seq_positions.append((i, valid_length))
    
    return sequences, seq_positions, stride

def griffin_lim(magnitude, n_iter=100, hop_length=None, win_length=None, window='hann', n_fft=None, length=None, momentum=0.99):
    """
    Griffin-Lim算法，用于从幅度谱重建相位
    
    Args:
        magnitude: 幅度谱 [F, T]
        n_iter: 迭代次数
        hop_length: 帧移
        win_length: 窗长
        window: 窗函数类型
        n_fft: FFT点数
        length: 输出信号长度
        momentum: 动量参数，加速收敛
        
    Returns:
        重建的信号
    """
    # 使用随机相位初始化
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    
    # 上一次的复数谱，用于动量
    last_spec = None
    
    # 初始化复数谱
    stft_matrix = magnitude * angles
    
    for i in range(n_iter):
        # 执行逆STFT
        y = librosa.istft(stft_matrix, hop_length=hop_length, win_length=win_length, window=window, length=length)
        
        # 再次执行STFT
        stft_matrix_new = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        
        # 使用动量加速收敛
        if momentum > 0 and last_spec is not None:
            stft_matrix_new = stft_matrix_new + momentum * (stft_matrix_new - last_spec)
        
        # 更新相位，保持幅度不变
        angles = np.exp(1j * np.angle(stft_matrix_new))
        last_spec = stft_matrix_new
        
        # 用新相位更新复数谱
        stft_matrix = magnitude * angles
        
        # 打印进度
        if (i + 1) % 20 == 0:
            print(f"Griffin-Lim迭代: {i + 1}/{n_iter}")
    
    # 最后一次逆STFT得到时域信号
    y = librosa.istft(stft_matrix, hop_length=hop_length, win_length=win_length, window=window, length=length)
    return y

def reconstruct_from_masks(mic_mag, mic_phase, masks, seq_positions, stride, config, overlap=False, use_griffin_lim=False, n_iter=50):
    """
    根据预测的掩码和麦克风信号重建增强后的语音
    
    Args:
        mic_mag: 麦克风信号的幅度谱，形状为 [F, T]，其中F是频点数，T是时间帧数
        mic_phase: 麦克风信号的相位谱
        masks: 预测的掩码，形状为 [n_sequences, seq_len, F]
        seq_positions: 每个序列在原始信号中的起始位置和有效长度
        stride: 重建时的步长
        config: 配置对象
        overlap: 是否使用重叠处理
        use_griffin_lim: 是否使用Griffin-Lim算法重建相位
        n_iter: Griffin-Lim算法的迭代次数
    """
    F, T = mic_mag.shape  # 频点数和时间帧数
    
    # 创建增强信号的幅度谱 [F, T]
    enhanced_mag = np.zeros_like(mic_mag)
    
    # 创建权重矩阵，用于处理重叠区域 [T]
    weights = np.zeros(T)
    
    print(f"麦克风幅度谱形状: {mic_mag.shape}, 掩码形状: {masks.shape}")
    
    # 对每个序列应用掩码
    for i, ((start_idx, valid_length), mask) in enumerate(zip(seq_positions, masks)):
        if valid_length <= 0:
            continue
            
        # 确保不超出范围
        valid_length = min(valid_length, mask.shape[0], T - start_idx)
        
        # 打印调试信息
        if i == 0:
            print(f"处理第一个序列 - 起始位置: {start_idx}, 有效长度: {valid_length}")
            print(f"当前掩码形状: {mask.shape}, 有效部分: {mask[:valid_length].shape}")
        
        # 计算当前序列的权重（用于重叠区域）
        if overlap:
            seq_weights = np.ones(valid_length)
            if i == 0:  # 第一个序列
                seq_weights[valid_length//2:] = np.linspace(1, 0, valid_length - valid_length//2)
            elif i == len(masks) - 1:  # 最后一个序列
                seq_weights[:valid_length//2] = np.linspace(0, 1, valid_length//2)
            else:  # 中间序列
                seq_weights[:valid_length//2] = np.linspace(0, 1, valid_length//2)
                seq_weights[valid_length//2:] = np.linspace(1, 0, valid_length - valid_length//2)
        else:
            seq_weights = np.ones(valid_length)
        
        # 遍历每个时间帧
        for t in range(valid_length):
            frame_idx = start_idx + t
            if frame_idx >= T:
                break
                
            # 获取当前帧的掩码 [F]
            frame_mask = mask[t, :F]
            
            # 应用掩码到幅度谱
            if overlap:
                enhanced_mag[:, frame_idx] += frame_mask * mic_mag[:, frame_idx] * seq_weights[t]
                weights[frame_idx] += seq_weights[t]
            else:
                enhanced_mag[:, frame_idx] = frame_mask * mic_mag[:, frame_idx]
                weights[frame_idx] = 1.0
    
    # 处理重叠区域，避免除零错误
    if overlap:
        weights = np.maximum(weights, 1e-8)
        for t in range(T):
            if weights[t] > 0:
                enhanced_mag[:, t] /= weights[t]
    
    # 根据选择使用不同的重建方法
    if use_griffin_lim:
        print("使用Griffin-Lim算法重建相位...")
        # 直接从增强的幅度谱使用Griffin-Lim算法重建信号
        enhanced_audio = griffin_lim(
            enhanced_mag,
            n_iter=n_iter,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=config.window,
            n_fft=config.n_fft
        )
    else:
        print("使用麦克风原始相位重建信号...")
        # 使用原始相位重建复数谱
        enhanced_stft = enhanced_mag * np.exp(1j * mic_phase)
        
        # 执行逆STFT得到时域信号
        enhanced_audio = librosa.istft(
            enhanced_stft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=config.window
        )
    
    return enhanced_audio

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查文件路径
    for path in [args.farend_path, args.mic_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
    
    if not os.path.exists(args.model_path):
        print(f"模型文件不存在: {args.model_path}")
        print(f"尝试搜索模型文件...")
        model_dir = os.path.dirname(args.model_path)
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if model_files:
                args.model_path = os.path.join(model_dir, model_files[-1])
                print(f"找到模型文件: {args.model_path}")
            else:
                raise FileNotFoundError(f"模型目录 {model_dir} 中没有找到模型文件")
    
    # 设置设备
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 加载配置
    config = Config()
    
    # 创建输出目录
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.model_path, config, device)
    
    # 加载音频文件
    print("加载音频文件...")
    farend_audio = load_and_resample(args.farend_path, config.sr)
    mic_audio = load_and_resample(args.mic_path, config.sr)
    
    # 确保两个信号长度相同
    min_length = min(len(farend_audio), len(mic_audio))
    farend_audio = farend_audio[:min_length]
    mic_audio = mic_audio[:min_length]
    
    
    # 预处理信号，提取特征
    print("预处理音频，提取特征...")
    features, mic_mag, mic_phase = preprocess_for_inference(farend_audio, mic_audio, config)
    
    print(f"特征形状: {features.shape}, 麦克风幅度谱形状: {mic_mag.shape}")
    
    # 拆分为多个序列
    print("将特征拆分为多个序列...")
    sequences, seq_positions, stride = split_into_sequences(features, config.max_seq_len, args.overlap)
    
    print(f"序列数量: {len(sequences)}, 序列形状: {sequences[0].shape}")
    
    # 创建一个批次
    batch = torch.FloatTensor(np.array(sequences)).to(device)
    
    # 推理
    print(f"执行模型推理，批次大小: {batch.shape}...")
    with torch.no_grad():
        # 处理大型批次时可能需要分块处理以避免内存不足
        batch_size = min(32, len(batch))
        all_outputs = []
        
        for i in range(0, len(batch), batch_size):
            batch_chunk = batch[i:i+batch_size]
            outputs_chunk = model(batch_chunk)
            all_outputs.append(outputs_chunk.cpu().numpy())
        
        # 合并所有批次的结果
        masks = np.concatenate(all_outputs, axis=0)
    
    print(f"掩码形状: {masks.shape}")
    
    # 根据掩码重建增强后的语音
    print("应用掩码，重建增强后的语音...")
    enhanced_audio = reconstruct_from_masks(
        mic_mag, mic_phase, masks, seq_positions, stride, config, 
        overlap=args.overlap, use_griffin_lim=args.griffin_lim, n_iter=args.n_iter
    )
    
    # 保存增强后的音频
    sf.write(args.output_path, enhanced_audio, config.sr)
    print(f"增强后的音频已保存至: {args.output_path}")
    
    # 计算信噪比提升（如果有目标信号）
    target_path = args.farend_path.replace('farend', 'target')
    if os.path.exists(target_path):
        target_audio = load_and_resample(target_path, config.sr)[:min_length]
        
        # 计算原始信噪比
        noise_before = mic_audio - target_audio
        snr_before = 10 * np.log10(np.sum(target_audio**2) / (np.sum(noise_before**2) + 1e-8))
        
        # 计算增强后信噪比
        noise_after = enhanced_audio - target_audio
        snr_after = 10 * np.log10(np.sum(target_audio**2) / (np.sum(noise_after**2) + 1e-8))
        
        print(f"原始信噪比: {snr_before:.2f} dB")
        print(f"增强后信噪比: {snr_after:.2f} dB")
        print(f"信噪比提升: {snr_after - snr_before:.2f} dB")

if __name__ == "__main__":
    main()