import librosa
import numpy as np
import scipy.signal

def _load_and_resample(path, target_sr):
    """加载音频并重采样到目标采样率。"""
    y, sr = librosa.load(path, sr=None, mono=True)  # y: (n_samples,)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)  # y: (n_resampled_samples,)
    return y

def _stft(y, frame_length, hop_length, win_length, window='hann'):
    """短时傅里叶变换。"""
    return librosa.stft(y, n_fft=frame_length, hop_length=hop_length,
                        win_length=win_length, window=window)  # (n_fft // 2 + 1, n_frames)

def _amplitude_phase(stft_matrix):
    """分离幅度和相位。"""
    return np.abs(stft_matrix), np.angle(stft_matrix)  # (n_fft // 2 + 1, n_frames), (n_fft // 2 + 1, n_frames)

def _log_power_spectrum(amplitude):
    """计算对数功率谱。"""
    return 20 * np.log10(np.maximum(amplitude, 1e-10))  # (n_fft // 2 + 1, n_frames)

def _prepare_features(farend, mic, target_sr, frame_length, hop_length, win_length, target=None):
    """
    执行通用的预处理步骤（加载、重采样、STFT、幅度相位分离、对数功率谱）。
    这个函数被 preprocess_audio 和 prepare_features_for_inference 共用。
    """
    # 1. 加载和重采样 (如果 target 为 None，则不加载 target)
    farend = _load_and_resample(farend, target_sr)  # farend: (n_farend_samples,)
    mic = _load_and_resample(mic, target_sr)      # mic: (n_mic_samples,)
    if target is not None:
        target = _load_and_resample(target, target_sr)  # target: (n_target_samples,)

    # 2. 对齐 (以最短的信号为准)
    if target is not None:
        min_length = min(len(farend), len(mic), len(target))
        farend = farend[:min_length]
        mic = mic[:min_length]
        target = target[:min_length]
    else:
        min_length = min(len(farend), len(mic))
        farend = farend[:min_length]
        mic = mic[:min_length]

    # 3. STFT
    farend_stft = _stft(farend, frame_length, hop_length, win_length)  # farend_stft: (n_fft // 2 + 1, n_frames)
    mic_stft = _stft(mic, frame_length, hop_length, win_length)        # mic_stft: (n_fft // 2 + 1, n_frames)
    if target is not None:
        target_stft = _stft(target, frame_length, hop_length, win_length)  # target_stft: (n_fft // 2 + 1, n_frames)

    # 4. 幅度谱和相位谱
    farend_amp, _ = _amplitude_phase(farend_stft)  # farend_amp: (n_fft // 2 + 1, n_frames)
    mic_amp, mic_phase = _amplitude_phase(mic_stft)  # mic_amp: (n_fft // 2 + 1, n_frames), mic_phase: (n_fft // 2 + 1, n_frames)
    if target is not None:
        target_amp, _ = _amplitude_phase(target_stft)  # target_amp: (n_fft // 2 + 1, n_frames)

    # 5. 特征计算：对数功率谱
    farend_log_power = _log_power_spectrum(farend_amp)  # farend_log_power: (n_fft // 2 + 1, n_frames)
    mic_log_power = _log_power_spectrum(mic_amp)      # mic_log_power: (n_fft // 2 + 1, n_frames)

    # 6. 特征拼接
    features = np.concatenate([mic_log_power, farend_log_power], axis=0).T  # features: (n_frames, 2 * (n_fft // 2 + 1))

    # 转置幅度谱和相位谱
    mic_amp = mic_amp.T  # (n_frames, 频率维度)
    mic_phase = mic_phase.T  # (n_frames, 频率维度)
    
    if target is not None:
        target_amp = target_amp.T  # (n_frames, 频率维度)
        return features, mic_phase, mic_amp, target_amp
    else:
        return features, mic_phase, mic_amp

def preprocess_audio(farend_path, mic_path, target_path, target_sr=16000,
                     frame_length=512, hop_length=256, win_length=512):
    """
    预处理单个音频文件对（用于训练），生成特征和目标掩码。
    """
    features, mic_phase, mic_amp, target_amp = _prepare_features(
        farend_path, mic_path, target_sr, frame_length, hop_length, win_length, target=target_path
    )

    # 7. 计算理想掩码 (IRM) - 修复：使用幅度谱而不是相位谱
    target_mask = np.clip(target_amp / (mic_amp + 1e-10), 0.0, 1.0)  # target_mask: (n_frames, n_fft // 2 + 1)
    num_frames = features.shape[0]

    return features, target_mask, num_frames, mic_phase

def prepare_features_for_inference(farend_path, mic_path, seq_len, target_sr=16000,
                                   frame_length=512, hop_length=256, win_length=512):
    """
    为模型推理准备输入特征。
    """
    features, mic_phase, mic_amp = _prepare_features(
        farend_path, mic_path, target_sr, frame_length, hop_length, win_length
    )
    num_frames = features.shape[0]
    
    # 7. 划分子序列 (滑动窗口)
    sub_sequences = []
    sub_phases = []
    sub_amps = []
    original_lengths = []

    start_frame = 0
    while start_frame + seq_len <= num_frames:
        end_frame = start_frame + seq_len
        sub_seq = features[start_frame:end_frame, :]       # sub_seq: (seq_len, 2 * (n_fft // 2 + 1))
        sub_phase = mic_phase[start_frame:end_frame, :]   # sub_phase: (seq_len, n_fft // 2 + 1)
        sub_amp = mic_amp[start_frame:end_frame, :]       # sub_amp: (seq_len, n_fft // 2 + 1)
        
        sub_sequences.append(sub_seq)
        sub_phases.append(sub_phase)
        sub_amps.append(sub_amp)
        original_lengths.append(seq_len)
        start_frame += hop_length

    # 8. 处理最后一个序列 (可能需要填充)
    if start_frame < num_frames:
        end_frame = num_frames
        sub_seq = features[start_frame:end_frame, :]  # sub_seq: (last_seq_len, 2 * (n_fft // 2 + 1))
        sub_phase = mic_phase[start_frame:end_frame, :]  # sub_phase: (last_seq_len, n_fft // 2 + 1)
        sub_amp = mic_amp[start_frame:end_frame, :]  # sub_amp: (last_seq_len, n_fft // 2 + 1)
        
        original_length = sub_seq.shape[0]
        pad_width = seq_len - original_length
        sub_seq = np.pad(sub_seq, ((0, pad_width), (0, 0)), 'constant')  # sub_seq: (seq_len, 2 * (n_fft // 2 + 1))
        sub_phase = np.pad(sub_phase, ((0, pad_width), (0, 0)), 'constant')  # sub_phase: (seq_len, n_fft // 2 + 1)
        sub_amp = np.pad(sub_amp, ((0, pad_width), (0, 0)), 'constant')  # sub_amp: (seq_len, n_fft // 2 + 1)
        
        sub_sequences.append(sub_seq)
        sub_phases.append(sub_phase)
        sub_amps.append(sub_amp)
        original_lengths.append(original_length)

    # 9. 转为 numpy 数组
    features_tensor = np.array(sub_sequences)   # features_tensor: (num_sequences, seq_len, 2 * (n_fft // 2 + 1))
    mic_phases_tensor = np.array(sub_phases)    # mic_phases_tensor: (num_sequences, seq_len, n_fft // 2 + 1)
    mic_amps_tensor = np.array(sub_amps)        # mic_amps_tensor: (num_sequences, seq_len, n_fft // 2 + 1)
     
    return features_tensor, mic_phases_tensor, mic_amps_tensor, num_frames, original_lengths

if __name__ == "__main__":
    # 训练数据预处理测试
    print("-" * 20)
    print("Testing preprocess_audio (for training):")
    train_farend_path = "f00000_farend.wav"  # 替换为你的训练远端音频路径
    train_mic_path = "f00000_mic.wav"        # 替换为你的训练麦克风音频路径
    train_target_path = "f00000_target.wav"    # 替换为你的训练目标音频路径

    try:
        features, target_mask, num_frames, mic_phase = preprocess_audio(
            train_farend_path, train_mic_path, train_target_path
        )

        print("Features shape:", features.shape)
        print("Target mask shape:", target_mask.shape)
        print("Number of frames:", num_frames)
        print("Mic phase shape:", mic_phase.shape)

        # 检查数值范围 (确保没有 nan 或 inf)
        assert np.isfinite(features).all(), "Features contain NaN or Inf values"
        assert np.isfinite(target_mask).all(), "Target mask contains NaN or Inf values"
        assert np.isfinite(mic_phase).all(), "Mic phase contains NaN or Inf values"

        print("preprocess_audio test passed!")

    except FileNotFoundError:
        print(f"Error: One or more training audio files not found.")
        print("Please make sure the following files exist:")
        print(f"- {train_farend_path}")
        print(f"- {train_mic_path}")
        print(f"- {train_target_path}")
    except Exception as e:
        print(f"An error occurred during preprocess_audio test: {e}")

    # 推理数据预处理测试
    print("\n" + "-" * 20)
    print("Testing prepare_features_for_inference (for inference):")
    inference_farend_path = "f00000_farend.wav"  # 替换为你的推理远端音频路径
    inference_mic_path = "f00000_target.wav"      # 替换为你的推理麦克风音频路径
    seq_len = 128  # 设置你模型期望的输入序列长度

    try:
        features_tensor, mic_phases_tensor, mic_amps_tensor, num_frames, original_lengths = prepare_features_for_inference(
            inference_farend_path, inference_mic_path, seq_len
        )

        print("Features tensor shape:", features_tensor.shape)
        print("Mic phases tensor shape:", mic_phases_tensor.shape)
        print("Mic amplitudes tensor shape:", mic_amps_tensor.shape)
        print("Original number of frames:", num_frames)
        print("Original lengths of sequences:", original_lengths)

        # 检查数值范围
        assert np.isfinite(features_tensor).all(), "Features tensor contains NaN or Inf values"
        assert np.isfinite(mic_phases_tensor).all(), "Mic phases tensor contains NaN or Inf values"
        assert np.isfinite(mic_amps_tensor).all(), "Mic amplitudes tensor contains NaN or Inf values"

        print("prepare_features_for_inference test passed!")

    except FileNotFoundError:
        print(f"Error: One or more inference audio files not found.")
        print("Please make sure the following files exist:")
        print(f"- {inference_farend_path}")
        print(f"- {inference_mic_path}")
    except Exception as e:
        print(f"An error occurred during prepare_features_for_inference test: {e}")