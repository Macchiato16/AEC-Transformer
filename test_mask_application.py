import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from preprocess import Config, load_and_resample, compute_stft, compute_log_power_spec, compute_ideal_mask
from inference import reconstruct_from_masks
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test_mask_application(mic_path, target_path, output_path, plot=True, use_griffin_lim=False, n_iter=50):
    """
    测试掩码应用函数
    
    Args:
        mic_path: 麦克风信号文件路径
        target_path: 目标信号文件路径
        output_path: 输出文件路径
        plot: 是否绘制频谱图对比
        use_griffin_lim: 是否使用Griffin-Lim算法重建相位
        n_iter: Griffin-Lim算法的迭代次数
    """
    # 加载配置
    config = Config()
    
    # 加载音频文件
    print(f"加载音频文件: {mic_path} 和 {target_path}")
    mic_audio = load_and_resample(mic_path, config.sr)
    target_audio = load_and_resample(target_path, config.sr)
    
    # 确保两个信号长度相同
    min_length = min(len(mic_audio), len(target_audio))
    mic_audio = mic_audio[:min_length]
    target_audio = target_audio[:min_length]
    
    print(f"音频长度: {min_length/config.sr:.2f}秒, 采样率: {config.sr}Hz")
    
    # 计算STFT
    print("计算STFT...")
    mic_mag, mic_phase = compute_stft(
        mic_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    target_mag, target_phase = compute_stft(
        target_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    
    # 计算理想掩码
    print("计算理想掩码...")
    ideal_mask = compute_ideal_mask(target_mag, mic_mag)
    
    # 计算幅度误差
    mag_error = np.mean(np.abs(ideal_mask * mic_mag - target_mag))
    print(f"理想掩码幅度误差: {mag_error:.6f}")
    
    # 将掩码转换为模型输出格式 [1, T, F]
    masks = np.expand_dims(ideal_mask.T, axis=0)
    
    # 准备序列位置信息（单个完整序列）
    seq_positions = [(0, masks.shape[1])]
    
    # 使用重建函数应用掩码
    print("应用掩码重建信号...")
    enhanced_audio = reconstruct_from_masks(
        mic_mag, mic_phase, masks, seq_positions, masks.shape[1], config, 
        overlap=False, use_griffin_lim=use_griffin_lim, n_iter=n_iter
    )
    
    # 计算重建后的信号与目标信号的归一化均方误差
    enhanced_audio = enhanced_audio[:min_length]  # 确保长度一致
    mse = np.mean((enhanced_audio - target_audio) ** 2) / np.mean(target_audio ** 2)
    print(f"重建信号与目标信号的归一化均方误差: {mse:.6f}")
    print(f"重建信号与目标信号的信噪比: {10 * np.log10(1 / mse):.2f} dB")
    
    # 保存增强后的音频
    sf.write(output_path, enhanced_audio, config.sr)
    print(f"重建信号已保存至: {output_path}")
    
    # 可选：绘制频谱图对比
    if plot:
        plot_comparison(mic_audio, target_audio, enhanced_audio, config, "comparison_plot.png")
    
    return mse

def plot_comparison(mic_audio, target_audio, enhanced_audio, config, save_path):
    """绘制三个信号的频谱图对比"""
    plt.figure(figsize=(15, 10))
    
    # 计算STFT并转换为dB
    def compute_db_spec(audio):
        spec = librosa.stft(audio, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length)
        return librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    
    mic_spec = compute_db_spec(mic_audio)
    target_spec = compute_db_spec(target_audio)
    enhanced_spec = compute_db_spec(enhanced_audio)
    
    # 绘制频谱图
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mic_spec, sr=config.sr, hop_length=config.hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('麦克风信号频谱图')
    
    plt.subplot(3, 1, 2)
    librosa.display.specshow(target_spec, sr=config.sr, hop_length=config.hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('目标信号频谱图')
    
    plt.subplot(3, 1, 3)
    librosa.display.specshow(enhanced_spec, sr=config.sr, hop_length=config.hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('重建信号频谱图')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"频谱图对比已保存至: {save_path}")
    
    # 绘制波形图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(mic_audio)
    plt.title('麦克风信号波形')
    plt.xlim([0, len(mic_audio)])
    
    plt.subplot(3, 1, 2)
    plt.plot(target_audio)
    plt.title('目标信号波形')
    plt.xlim([0, len(target_audio)])
    
    plt.subplot(3, 1, 3)
    plt.plot(enhanced_audio)
    plt.title('重建信号波形')
    plt.xlim([0, len(enhanced_audio)])
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_waveform.png'))
    plt.close()
    print(f"波形图对比已保存至: {save_path.replace('.png', '_waveform.png')}")
    
    # 绘制STFT相位差
    mic_phase = np.angle(librosa.stft(mic_audio, n_fft=config.n_fft, hop_length=config.hop_length))
    target_phase = np.angle(librosa.stft(target_audio, n_fft=config.n_fft, hop_length=config.hop_length))
    enhanced_phase = np.angle(librosa.stft(enhanced_audio, n_fft=config.n_fft, hop_length=config.hop_length))
    
    # 计算相位差
    phase_diff_original = np.abs(np.angle(np.exp(1j * (target_phase - mic_phase))))
    phase_diff_enhanced = np.abs(np.angle(np.exp(1j * (target_phase - enhanced_phase))))
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(phase_diff_original, sr=config.sr, hop_length=config.hop_length, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title('原始信号与目标信号的相位差')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(phase_diff_enhanced, sr=config.sr, hop_length=config.hop_length, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title('重建信号与目标信号的相位差')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_phase.png'))
    plt.close()
    print(f"相位差图已保存至: {save_path.replace('.png', '_phase.png')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试掩码应用函数')
    parser.add_argument('--mic', type=str, required=True, help='麦克风信号文件路径')
    parser.add_argument('--target', type=str, required=True, help='目标信号文件路径')
    parser.add_argument('--output', type=str, default='enhanced_test.wav', help='输出文件路径')
    parser.add_argument('--no-plot', action='store_true', help='不绘制频谱图')
    parser.add_argument('--griffin-lim', action='store_true', help='使用Griffin-Lim算法重建相位')
    parser.add_argument('--n-iter', type=int, default=50, help='Griffin-Lim算法的迭代次数')
    parser.add_argument('--compare', action='store_true', help='生成两种方法的对比结果')
    
    args = parser.parse_args()
    
    if args.compare:
        # 对比不同重建方法
        print("\n===== 使用原始相位重建 =====")
        test_mask_application(args.mic, args.target, args.output.replace('.wav', '_original_phase.wav'), 
                             not args.no_plot, use_griffin_lim=False)
        
        print("\n===== 使用Griffin-Lim算法重建 =====")
        test_mask_application(args.mic, args.target, args.output.replace('.wav', '_griffin_lim.wav'), 
                             not args.no_plot, use_griffin_lim=True, n_iter=args.n_iter)
    else:
        # 使用单一重建方法
        test_mask_application(args.mic, args.target, args.output, not args.no_plot, 
                             use_griffin_lim=args.griffin_lim, n_iter=args.n_iter)