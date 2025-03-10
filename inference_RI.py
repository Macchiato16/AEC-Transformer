import os
import torch
import librosa
import soundfile as sf
import numpy as np
from preprocess_RI import Config, load_and_resample, compute_stft, trim_or_pad
from model.transfomer import TransformerEchoCancellation
import argparse
import time
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def parse_args():
    parser = argparse.ArgumentParser(description='使用Transformer模型进行回声消除推理')
    parser.add_argument('--farend_path', type=str, default='f00000_farend.wav', help='远端信号音频文件路径')
    parser.add_argument('--mic_path', type=str, default='f00000_mic.wav', help='麦克风信号音频文件路径')
    parser.add_argument('--model_path', type=str, default='checkpoints_RI/model_final.pt', help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='output', help='增强后音频输出目录')
    parser.add_argument('--output_filename', type=str, default='enhanced.wav', help='增强后音频文件名')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='使用CUDA进行推理')
    parser.add_argument('--plot_waveform', action='store_true', default=True, help='是否绘制波形图')
    return parser.parse_args()

def create_model(config, model_path, device):
    """创建模型并加载预训练权重"""
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
    model.to(device)

    # 加载预训练模型
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    model.eval()  # 设置为评估模式
    return model

def preprocess_inference_audio(farend_path, mic_path, config):
    """预处理用于推理的音频文件"""
    # 加载并重采样音频
    farend_audio = load_and_resample(farend_path, config.sr)
    mic_audio = load_and_resample(mic_path, config.sr)

    if farend_audio is None or mic_audio is None:
        raise ValueError("加载音频文件失败，请检查文件路径")

    # 获取最短长度并裁剪
    farend_audio, mic_audio = trim_or_pad([farend_audio, mic_audio])

    # 计算STFT
    farend_stft = compute_stft(
        farend_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    mic_stft = compute_stft(
        mic_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )

    # 提取实部和虚部
    farend_real = np.real(farend_stft).T  # [T, F]
    farend_imag = np.imag(farend_stft).T  # [T, F]
    mic_real = np.real(mic_stft).T        # [T, F]
    mic_imag = np.imag(mic_stft).T        # [T, F]

    # 拼接特征
    features = np.concatenate([mic_real, mic_imag, farend_real, farend_imag], axis=1) # [T, F*4]

    if features.shape[0] > config.max_seq_len:
        features = features[:config.max_seq_len, :]

    return features

def reconstruct_audio(real_part, imag_part, config):
    """从实部和虚部重建音频"""
    stft = real_part.T + 1j * imag_part.T # 转置回 [F, T]
    enhanced_audio = librosa.istft(
        stft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=config.window
    )
    return enhanced_audio

def plot_waveform(audio, sr, ax, title):
    """在给定的 axes 对象上绘制音频时域波形图"""
    time = np.arange(0, len(audio)) / sr
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

def process_inference(farend_path, mic_path, model, config, device, output_dir, output_filename, plot_waveform_flag=False):
    """执行推理过程"""
    # 加载原始音频数据用于绘制波形图
    farend_audio_waveform, _ = librosa.load(farend_path, sr=config.sr)
    mic_audio_waveform, _ = librosa.load(mic_path, sr=config.sr)

    # 预处理音频
    features = preprocess_inference_audio(farend_path, mic_path, config)
    feature_dim = features.shape[-1]
    seq_len = features.shape[0]

    # 转换为torch张量并移至设备
    features_torch = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device) # [1, T, F*4]

    start_time = time.time()
    # 执行推理
    with torch.no_grad():
        predictions = model(features_torch) # [1, T, F*2]

    inference_time = time.time() - start_time
    print(f"推理耗时: {inference_time:.4f} 秒，音频长度: {seq_len * config.hop_length / config.sr:.2f} 秒")

    # 将预测结果移至CPU并转换为numpy
    predictions_np = predictions.squeeze(0).cpu().numpy() # [T, F*2]

    # 分离实部和虚部
    target_real, target_imag = np.split(predictions_np, 2, axis=-1) # [T, F], [T, F]

    # 重建音频
    enhanced_audio = reconstruct_audio(target_real, target_imag, config)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # 保存增强后的音频
    sf.write(output_path, enhanced_audio, config.sr)
    print(f"增强后的音频保存至: {output_path}")

    if plot_waveform_flag:
        plot_filename = os.path.splitext(output_filename)[0] + '_waveforms' # 波形图文件名 (复数)
        fig, ax = plt.subplots(3, 1, figsize=(10, 8)) # 创建包含 3 个子图的 figure 和 axes

        # 绘制远端信号波形图
        plot_waveform(farend_audio_waveform, config.sr, ax[0], title='Farend Audio Waveform')

        # 绘制麦克风信号波形图
        plot_waveform(mic_audio_waveform, config.sr, ax[1], title='Mic Audio Waveform')

        # 绘制增强后音频波形图
        plot_waveform(enhanced_audio, config.sr, ax[2], title='Enhanced Audio Waveform')

        plt.tight_layout() # 调整子图布局，避免重叠
        plot_path = os.path.join(output_dir, plot_filename) + '.png'
        plt.savefig(plot_path) # 保存整个 figure
        plt.close()
        print(f"波形图保存至: {plot_path}")

def main():
    args = parse_args()

    # 配置
    config = Config()

    # 设备
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")

    # 创建模型并加载权重
    model = create_model(config, args.model_path, device)

    # 执行推理
    process_inference(
        args.farend_path,
        args.mic_path,
        model,
        config,
        device,
        args.output_dir,
        args.output_filename,
        plot_waveform_flag=args.plot_waveform
    )

if __name__ == '__main__':
    main()
