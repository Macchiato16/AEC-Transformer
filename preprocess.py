import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from tqdm import tqdm
import h5py
import warnings
warnings.filterwarnings('ignore')

# 参数设置
class Config:
    def __init__(self):
        # 基本参数
        self.sr = 16000  # 目标采样率
        self.duration = 10  # 每个音频片段的长度（秒）
        self.n_fft = 320  # FFT点数
        self.hop_length = 160  # 帧移
        self.win_length = 320  # 窗长
        self.window = 'hann'  # 窗函数类型
        
        # 数据集参数
        self.root_dir = 'data/synthetic/'  # 数据集根目录
        self.output_dir = 'data/preprocessed/'  # 处理后数据保存目录
        self.meta_file = 'data/meta.csv'  # 元数据文件
        
        # Transformer模型参数
        self.max_seq_len = 50  # Transformer输入的最大序列长度
        self.feature_dim = self.n_fft // 2 + 1  # 特征维度（STFT后的频点数）
        
        # 数据集分割
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        
        # 训练参数
        self.batch_size = 32
        self.num_workers = min(multiprocessing.cpu_count(), 8)
        
        # 处理参数
        self.min_db = -80  # 对数谱的最小dB值
        self.ref_db = 20  # 对数谱的参考dB值

# 音频处理函数
def load_and_resample(file_path, sr=16000):
    """加载并重采样音频"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio

def get_min_length(audios):
    """获取多个音频的最小长度"""
    lengths = [len(audio) for audio in audios if audio is not None]
    return min(lengths) if lengths else 0

def trim_or_pad(audios, target_length=None):
    """根据最小长度修剪多个音频"""
    if not audios or all(audio is None for audio in audios):
        return [None] * len(audios)
    
    # 如果没有指定目标长度，则使用最小长度
    if target_length is None:
        target_length = get_min_length(audios)
    
    result = []
    for audio in audios:
        if audio is None:
            result.append(None)
        elif len(audio) > target_length:
            result.append(audio[:target_length])
        else:
            result.append(np.pad(audio, (0, target_length - len(audio)), 'constant'))
    
    return result

def compute_stft(audio, n_fft=512, hop_length=256, win_length=512, window='hann'):
    """计算短时傅里叶变换"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    magnitude, phase = librosa.magphase(stft)
    return magnitude, phase

def compute_log_power_spec(magnitude, min_db=-80, ref_db=20):
    """计算对数功率谱"""
    power = np.abs(magnitude) ** 2
    log_power = librosa.power_to_db(power, ref=ref_db, top_db=-min_db)
    # 归一化到[0, 1]范围
    log_power = (log_power - min_db) / (-min_db)
    log_power = np.clip(log_power, 0, 1)
    return log_power

def compute_ideal_mask(target_mag, mic_mag, eps=1e-8):
    """计算理想振幅掩码 (IAM)"""
    mask = target_mag / (mic_mag + eps)
    mask = np.clip(mask, 0, 1)  # 限制掩码范围在[0, 1]
    return mask

def preprocess_audio_files(file_id, config):
    """预处理单个音频文件组"""
    # 构建文件路径
    farend_path = os.path.join(config.root_dir, f'{file_id}_farend.wav')
    mic_path = os.path.join(config.root_dir, f'{file_id}_mic.wav')
    target_path = os.path.join(config.root_dir, f'{file_id}_target.wav')
    
    # 加载并重采样音频
    farend_audio = load_and_resample(farend_path, config.sr)
    mic_audio = load_and_resample(mic_path, config.sr)
    target_audio = load_and_resample(target_path, config.sr)
    
    if farend_audio is None or mic_audio is None or target_audio is None:
        return None
    
    # 获取三个信号的最短长度并据此裁剪
    # 不使用预设的持续时间，而是使用实际最短长度
    farend_audio, mic_audio, target_audio = trim_or_pad([farend_audio, mic_audio, target_audio])
    
    # 计算STFT
    farend_mag, farend_phase = compute_stft(
        farend_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    mic_mag, mic_phase = compute_stft(
        mic_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    target_mag, target_phase = compute_stft(
        target_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    
    # 计算对数功率谱
    farend_log_power = compute_log_power_spec(farend_mag, config.min_db, config.ref_db)
    mic_log_power = compute_log_power_spec(mic_mag, config.min_db, config.ref_db)
    
    # 拼接特征（麦克风信号和远端信号的对数功率谱）
    # 特征形状: [T, F*2]，其中T是时间帧数，F是频点数
    features = np.concatenate([mic_log_power.T, farend_log_power.T], axis=1)
    
    # 计算理想掩码
    ideal_mask = compute_ideal_mask(target_mag, mic_mag)
    ideal_mask = ideal_mask.T  # 转置为[T, F]形状
    
    return {
        'file_id': file_id,
        'features': features,  # [T, F*2]
        'masks': ideal_mask,   # [T, F]
        'mic_phase': mic_phase # 保存相位信息，用于语音重建
    }

def process_dataset(config):
    """处理整个数据集并保存为HDF5格式"""
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 读取元数据
    meta_df = pd.read_csv(config.meta_file)
    file_ids = meta_df['file_id'].tolist()
    
    # 对每个文件进行预处理
    processed_data = []
    for file_id in tqdm(file_ids, desc="处理音频文件"):
        result = preprocess_audio_files(file_id, config)
        if result is not None:
            processed_data.append(result)
    
    # 随机打乱数据
    np.random.shuffle(processed_data)
    
    # 划分训练、验证和测试集
    n_samples = len(processed_data)
    n_train = int(config.train_ratio * n_samples)
    n_val = int(config.val_ratio * n_samples)
    
    train_data = processed_data[:n_train]
    val_data = processed_data[n_train:n_train+n_val]
    test_data = processed_data[n_train+n_val:]
    
    # 保存为HDF5格式
    save_to_hdf5(train_data, os.path.join(config.output_dir, 'train.h5'), config)
    save_to_hdf5(val_data, os.path.join(config.output_dir, 'val.h5'), config)
    save_to_hdf5(test_data, os.path.join(config.output_dir, 'test.h5'), config)
    
    print(f"数据处理完成！训练集: {len(train_data)}，验证集: {len(val_data)}，测试集: {len(test_data)}")

def save_to_hdf5(data_list, output_path, config):
    """将处理后的数据保存为HDF5格式"""
    with h5py.File(output_path, 'w') as f:
        for i, item in enumerate(data_list):
            group = f.create_group(f'sample_{i}')
            group.attrs['file_id'] = item['file_id']
            
            # 存储特征和掩码
            group.create_dataset('features', data=item['features'], compression='gzip')
            group.create_dataset('masks', data=item['masks'], compression='gzip')
            
            # 可选：存储相位信息（用于语音重建）
            if 'mic_phase' in item:
                group.create_dataset('mic_phase', data=item['mic_phase'], compression='gzip')
                
        # 存储配置信息
        config_group = f.create_group('config')
        config_group.attrs['sr'] = config.sr
        config_group.attrs['n_fft'] = config.n_fft
        config_group.attrs['hop_length'] = config.hop_length
        config_group.attrs['win_length'] = config.win_length
        config_group.attrs['feature_dim'] = config.feature_dim
        config_group.attrs['max_seq_len'] = config.max_seq_len

# 数据集类
class AECDataset(Dataset):
    def __init__(self, h5_path, max_seq_len=128, stride=64, transform=None):
        """
        声学回声消除数据集
        
        参数:
            h5_path (str): HDF5文件路径
            max_seq_len (int): 最大序列长度（Transformer输入的帧数）
            stride (int): 滑动窗口的步长（帧数）
            transform (callable, optional): 用于特征和标签的可选变换
        """
        self.h5_path = h5_path
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.transform = transform
        
        # 在 __init__ 中打开 HDF5 文件
        self.h5_file = h5py.File(h5_path, 'r')
        self.config = self.h5_file['config']
        self.feature_dim = self.config.attrs['feature_dim']

        self.sample_indices = []
        for i in range(len(self.h5_file.keys()) - 1):
            sample_key = f'sample_{i}'
            if sample_key in self.h5_file:
                features = self.h5_file[sample_key]['features']
                num_frames = features.shape[0]
                for start_idx in range(0, num_frames - max_seq_len + 1, stride):
                    self.sample_indices.append((i, start_idx))
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        sample_idx, start_frame = self.sample_indices[idx]
        sample_key = f'sample_{sample_idx}'

        # 直接从已打开的 HDF5 文件中读取数据
        features = self.h5_file[sample_key]['features'][start_frame:start_frame + self.max_seq_len]
        masks = self.h5_file[sample_key]['masks'][start_frame:start_frame + self.max_seq_len]

        if self.transform:
            features, masks = self.transform(features, masks)

        return torch.FloatTensor(features), torch.FloatTensor(masks)
    def __del__(self):
        # 在对象销毁时关闭 HDF5 文件
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

# 带有缓存功能的数据集类（用于训练加速）
class CachedAECDataset(Dataset):
    def __init__(self, h5_path, max_seq_len=128, stride=64, transform=None, cache_size=2000):
        """
        带缓存功能的声学回声消除数据集
        
        参数:
            h5_path (str): HDF5文件路径
            max_seq_len (int): 最大序列长度
            stride (int): 滑动窗口的步长
            transform (callable, optional): 用于特征和标签的可选变换
            cache_size (int): 缓存大小（样本数）
        """
        self.dataset = AECDataset(h5_path, max_seq_len, stride, transform)
        self.cache_size = min(cache_size, len(self.dataset))
        self.cache = {}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        item = self.dataset[idx]
        
        # 如果缓存未满，加入缓存
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
        
        return item

# 主函数
def main():
    config = Config()
    process_dataset(config)
    
    # 创建数据加载器示例
    train_dataset = CachedAECDataset(
        os.path.join(config.output_dir, 'train.h5'),
        max_seq_len=config.max_seq_len,
        stride=config.max_seq_len // 2  # 50%重叠
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"数据集创建完成！训练集大小: {len(train_dataset)}")
    
    # 获取一个批次的数据，检查形状
    for features, masks in train_loader:
        print(f"特征形状: {features.shape}, 掩码形状: {masks.shape}")
        break

if __name__ == "__main__":
    main()