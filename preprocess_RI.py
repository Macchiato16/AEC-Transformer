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
        self.output_dir = 'data/preprocessed_RI/'  # 处理后数据保存目录
        self.meta_file = 'data/meta.csv'  # 元数据文件
        
        # Transformer模型参数
        self.max_seq_len = 999  # Transformer输入的最大序列长度
        self.feature_dim = (self.n_fft // 2 + 1) * 2  # 特征维度（STFT后的频点数*2，实部+虚部）
        
        # 数据集分割
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        
        # 训练参数
        self.batch_size = 32
        self.num_workers = min(multiprocessing.cpu_count(), 8)

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
    """计算短时傅里叶变换并返回复数谱"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    return stft

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
    farend_audio, mic_audio, target_audio = trim_or_pad([farend_audio, mic_audio, target_audio])
    
    # 计算STFT（保留复数形式）
    farend_stft = compute_stft(
        farend_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    mic_stft = compute_stft(
        mic_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    target_stft = compute_stft(
        target_audio, config.n_fft, config.hop_length, config.win_length, config.window
    )
    
    # 提取实部和虚部
    farend_real = np.real(farend_stft)
    farend_imag = np.imag(farend_stft)
    mic_real = np.real(mic_stft)
    mic_imag = np.imag(mic_stft)
    target_real = np.real(target_stft)
    target_imag = np.imag(target_stft)
    
    # 转置为[T, F]形状
    farend_real = farend_real.T
    farend_imag = farend_imag.T
    mic_real = mic_real.T
    mic_imag = mic_imag.T
    target_real = target_real.T
    target_imag = target_imag.T
    
    # 拼接特征：[麦克风信号实部, 麦克风信号虚部, 远端信号实部, 远端信号虚部]
    # 特征形状: [T, F*4]，其中T是时间帧数，F是频点数
    features = np.concatenate([mic_real, mic_imag, farend_real, farend_imag], axis=1)
    
    # 拼接目标：[目标信号实部, 目标信号虚部]
    # 目标形状: [T, F*2]
    targets = np.concatenate([target_real, target_imag], axis=1)
    
    return {
        'file_id': file_id,
        'features': features,  # [T, F*4]
        'targets': targets,    # [T, F*2]
    }

def process_dataset(config):
    """处理整个数据集并分块保存为HDF5格式"""
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
    
    # 随机打乱数据前，先创建索引文件
    file_id_mapping = {i: data['file_id'] for i, data in enumerate(processed_data)}
    with open(os.path.join(config.output_dir, 'file_id_mapping.json'), 'w') as f:
        import json
        json.dump(file_id_mapping, f)
    
    # 随机打乱数据
    np.random.shuffle(processed_data)
    
    # 划分训练、验证和测试集
    n_samples = len(processed_data)
    n_train = int(config.train_ratio * n_samples)
    n_val = int(config.val_ratio * n_samples)
    
    train_data = processed_data[:n_train]
    val_data = processed_data[n_train:n_train+n_val]
    test_data = processed_data[n_train+n_val:]
    
    # 分块大小（每个块包含的样本数）
    chunk_size = 500  # 调整此值以适应你的内存限制
    
    # 保存训练集（分块）
    train_chunks = [train_data[i:i+chunk_size] for i in range(0, len(train_data), chunk_size)]
    for i, chunk in enumerate(train_chunks):
        chunk_name = f'train_chunk_{i}.h5'
        save_to_hdf5(chunk, os.path.join(config.output_dir, chunk_name), config)
        print(f"保存训练集块 {i+1}/{len(train_chunks)}, 大小: {len(chunk)}")
    
    # 保存验证集（分块）
    val_chunks = [val_data[i:i+chunk_size] for i in range(0, len(val_data), chunk_size)]
    for i, chunk in enumerate(val_chunks):
        chunk_name = f'val_chunk_{i}.h5'
        save_to_hdf5(chunk, os.path.join(config.output_dir, chunk_name), config)
        print(f"保存验证集块 {i+1}/{len(val_chunks)}, 大小: {len(chunk)}")
    
    # 保存测试集（分块）
    test_chunks = [test_data[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]
    for i, chunk in enumerate(test_chunks):
        chunk_name = f'test_chunk_{i}.h5'
        save_to_hdf5(chunk, os.path.join(config.output_dir, chunk_name), config)
        print(f"保存测试集块 {i+1}/{len(test_chunks)}, 大小: {len(chunk)}")
    
    # 创建索引文件
    dataset_info = {
        'train': {
            'total_samples': len(train_data),
            'chunks': len(train_chunks),
            'chunk_files': [f'train_chunk_{i}.h5' for i in range(len(train_chunks))]
        },
        'val': {
            'total_samples': len(val_data),
            'chunks': len(val_chunks),
            'chunk_files': [f'val_chunk_{i}.h5' for i in range(len(val_chunks))]
        },
        'test': {
            'total_samples': len(test_data),
            'chunks': len(test_chunks),
            'chunk_files': [f'test_chunk_{i}.h5' for i in range(len(test_chunks))]
        }
    }
    
    with open(os.path.join(config.output_dir, 'dataset_index.json'), 'w') as f:
        import json
        json.dump(dataset_info, f, indent=2)
    
    print(f"数据处理完成！\n训练集: {len(train_data)} 样本，分为 {len(train_chunks)} 块\n验证集: {len(val_data)} 样本，分为 {len(val_chunks)} 块\n测试集: {len(test_data)} 样本，分为 {len(test_chunks)} 块")

def save_to_hdf5(data_list, output_path, config):
    """将处理后的数据保存为HDF5格式"""
    with h5py.File(output_path, 'w') as f:
        for i, item in enumerate(data_list):
            group = f.create_group(f'sample_{i}')
            group.attrs['file_id'] = item['file_id']
            
            # 存储特征和目标
            group.create_dataset('features', data=item['features'], compression='gzip')
            group.create_dataset('targets', data=item['targets'], compression='gzip')
                
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
    def __init__(self, h5_dir, dataset_type='train', max_seq_len=128, stride=64, transform=None):
        """
        声学回声消除数据集 - 支持分块H5文件
        
        参数:
            h5_dir (str): 包含HDF5文件的目录路径
            dataset_type (str): 数据集类型，'train', 'val' 或 'test'
            max_seq_len (int): 最大序列长度（Transformer输入的帧数）
            stride (int): 滑动窗口的步长（帧数）
            transform (callable, optional): 用于特征和标签的可选变换
        """
        self.h5_dir = h5_dir
        self.dataset_type = dataset_type
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.transform = transform
        
        # 加载数据集索引
        import json
        index_path = os.path.join(h5_dir, 'dataset_index.json')
        with open(index_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        # 获取配置信息
        first_chunk = self.dataset_info[dataset_type]['chunk_files'][0]
        with h5py.File(os.path.join(h5_dir, first_chunk), 'r') as f:
            self.config = dict(f['config'].attrs)
            self.feature_dim = self.config['feature_dim']
        
        # 构建样本索引
        self.chunk_files = self.dataset_info[dataset_type]['chunk_files']
        self.chunk_sample_counts = []  # 每个块中的样本数
        self.sample_indices = []       # (chunk_idx, sample_idx, start_frame)
        
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with h5py.File(os.path.join(h5_dir, chunk_file), 'r') as f:
                samples_in_chunk = len(f.keys()) - 1  # 减去配置组
                self.chunk_sample_counts.append(samples_in_chunk)
                
                # 为每个样本的每个可能的时间段创建索引
                for sample_idx in range(samples_in_chunk):
                    sample_key = f'sample_{sample_idx}'
                    if sample_key in f:
                        features = f[sample_key]['features']
                        num_frames = features.shape[0]
                        for start_idx in range(0, num_frames - max_seq_len + 1, stride):
                            self.sample_indices.append((chunk_idx, sample_idx, start_idx))
        
        # 缓存已打开的H5文件
        self.h5_files = {}
    
    def __len__(self):
        return len(self.sample_indices)
    
    def _get_h5_file(self, chunk_idx):
        """获取H5文件对象，如果没有打开则打开它"""
        if chunk_idx not in self.h5_files:
            # 限制打开的文件数量，避免"打开文件过多"错误
            if len(self.h5_files) >= 10:  # 保持最多10个文件打开
                # 关闭最早打开的文件
                oldest_key = next(iter(self.h5_files))
                self.h5_files[oldest_key].close()
                del self.h5_files[oldest_key]
            
            # 打开新文件
            file_path = os.path.join(self.h5_dir, self.chunk_files[chunk_idx])
            self.h5_files[chunk_idx] = h5py.File(file_path, 'r')
        
        return self.h5_files[chunk_idx]
    
    def __getitem__(self, idx):
        chunk_idx, sample_idx, start_frame = self.sample_indices[idx]
        
        # 获取对应的H5文件
        h5_file = self._get_h5_file(chunk_idx)
        
        # 读取数据
        sample_key = f'sample_{sample_idx}'
        features = h5_file[sample_key]['features'][start_frame:start_frame + self.max_seq_len]
        targets = h5_file[sample_key]['targets'][start_frame:start_frame + self.max_seq_len]

        if self.transform:
            features, targets = self.transform(features, targets)

        return torch.FloatTensor(features), torch.FloatTensor(targets)
    
    def __del__(self):
        # 在对象销毁时关闭所有H5文件
        for h5_file in self.h5_files.values():
            h5_file.close()

# 带有缓存功能的数据集类（用于训练加速）
class CachedAECDataset(Dataset):
    def __init__(self, h5_dir, dataset_type='train', max_seq_len=128, stride=64, transform=None, cache_size=2000):
        """
        带缓存功能的声学回声消除数据集
        
        参数:
            h5_dir (str): 包含HDF5文件的目录路径
            dataset_type (str): 数据集类型，'train', 'val' 或 'test'
            max_seq_len (int): 最大序列长度
            stride (int): 滑动窗口的步长
            transform (callable, optional): 用于特征和标签的可选变换
            cache_size (int): 缓存大小（样本数）
        """
        self.dataset = AECDataset(h5_dir, dataset_type, max_seq_len, stride, transform)
        self.cache_size = min(cache_size, len(self.dataset))
        self.cache = {}
        self.cache_hit_count = 0
        self.total_access_count = 0
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        self.total_access_count += 1
        
        if idx in self.cache:
            self.cache_hit_count += 1
            return self.cache[idx]
        
        item = self.dataset[idx]
        
        # 如果缓存未满，加入缓存
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
        # 可选：实现LRU（最近最少使用）缓存替换策略
        # 但简单起见，这里不实现
        
        # 每1000次访问打印一次缓存命中率
        if self.total_access_count % 1000 == 0:
            hit_rate = self.cache_hit_count / self.total_access_count * 100
            print(f"缓存命中率: {hit_rate:.2f}% ({self.cache_hit_count}/{self.total_access_count})")
        
        return item

# 主函数
def main():
    config = Config()
    process_dataset(config)
    
    # 创建数据加载器示例
    train_dataset = CachedAECDataset(
        config.output_dir,  # 现在传入的是目录路径，而不是单个文件
        dataset_type='train',
        max_seq_len=config.max_seq_len,
        stride=config.max_seq_len // 2,  # 50%重叠
        cache_size=1000  # 减小缓存大小以适应内存限制
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
    for features, targets in train_loader:
        print(f"特征形状: {features.shape}, 目标形状: {targets.shape}")
        break
    
    # 创建并测试验证集加载器
    val_dataset = CachedAECDataset(
        config.output_dir,
        dataset_type='val',
        max_seq_len=config.max_seq_len,
        stride=config.max_seq_len  # 验证集可以使用不重叠的窗口
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"验证集大小: {len(val_dataset)}")
    
    # 打印数据集信息
    print("\n数据集分块信息:")
    import json
    with open(os.path.join(config.output_dir, 'dataset_index.json'), 'r') as f:
        dataset_info = json.load(f)
        
    for dataset_type, info in dataset_info.items():
        print(f"{dataset_type}: {info['total_samples']} 样本, {info['chunks']} 个块")

if __name__ == "__main__":
    main()