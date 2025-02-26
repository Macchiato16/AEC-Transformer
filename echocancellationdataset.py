import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from preprocess import preprocess_audio, prepare_features_for_inference
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

class EchoCancellationDataset(Dataset):
    """
    回声消除数据集类
    
    该类负责加载和预处理回声消除数据集，将音频文件转换为模型可用的特征。
    它支持将长音频序列分割为固定长度的子序列，以便于批处理训练。
    """
    
    def __init__(self, data_dir, meta_file, seq_len=50, target_sr=16000,
                 frame_length=320, hop_length=160, win_length=320, 
                 mode='train', cache_features=False):
        """
        初始化回声消除数据集
        
        参数:
            data_dir (str): 数据目录路径，包含所有音频文件
            meta_file (str): 元数据CSV文件路径，包含文件ID和属性信息
            seq_len (int): 序列长度（帧数），设置为50帧，表示每个样本包含的时间帧数
            target_sr (int): 目标采样率，设置为16000Hz，所有音频将重采样到此采样率
            frame_length (int): STFT帧长，设置为320点（对应16kHz采样率下的20ms）
            hop_length (int): STFT帧移，设置为160点（对应10ms，50%重叠）
            win_length (int): STFT窗长，设置为320点（通常等于帧长）
            mode (str): 'train'（训练模式）或'inference'（推理模式）
            cache_features (bool): 是否缓存特征（适用于小数据集，可减少I/O开销）
        """
        self.data_dir = data_dir
        self.meta_df = pd.read_csv(meta_file)
        self.seq_len = seq_len
        self.target_sr = target_sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.mode = mode
        self.cache_features = cache_features
        
        # 存储每个文件的子序列信息
        # 对于训练模式：(file_id, start_frame, end_frame)
        # 对于推理模式：(file_id, seq_idx)
        self.sample_indices = []
        
        # 如果启用缓存，创建缓存字典
        # 键：file_id，值：预处理后的特征
        self.feature_cache = {} if cache_features else None
        
        # 预处理元数据，构建样本索引
        self._preprocess_metadata()
    
    def _preprocess_metadata(self):
        """
        预处理元数据，构建样本索引
        
        该方法遍历元数据中的所有文件，计算每个文件可以生成多少个子序列，
        并将这些子序列的索引信息存储在sample_indices列表中。
        """
        print(f"预处理元数据，共 {len(self.meta_df)} 个文件...")
        
        # 使用tqdm显示进度条
        for idx, row in tqdm(self.meta_df.iterrows(), total=len(self.meta_df)):
            file_id = row['file_id']
            
            # 构建文件路径
            farend_path = os.path.join(self.data_dir, f"{file_id}_farend.wav")
            mic_path = os.path.join(self.data_dir, f"{file_id}_mic.wav")
            
            if self.mode == 'train':
                # 训练模式需要目标信号
                target_path = os.path.join(self.data_dir, f"{file_id}_target.wav")
                
                # 获取特征信息（不加载实际数据，只获取帧数）
                _, _, num_frames, _ = preprocess_audio(
                    farend_path, mic_path, target_path,
                    self.target_sr, self.frame_length, self.hop_length, self.win_length
                )
                
                # 计算可以提取的完整子序列数量
                # 使用滑动窗口方法，每次移动hop_length帧
                num_sequences = (num_frames - self.seq_len) // self.hop_length + 1
                if num_frames <= self.seq_len:
                    # 如果总帧数小于序列长度，则只生成一个子序列
                    num_sequences = 1
                
                # 为每个子序列添加索引
                for seq_idx in range(num_sequences):
                    start_frame = seq_idx * self.hop_length
                    end_frame = start_frame + self.seq_len
                    
                    # 处理最后一个可能不完整的序列
                    if end_frame > num_frames:
                        if num_frames - start_frame < self.seq_len // 2:
                            # 如果剩余帧数太少（小于序列长度的一半），跳过
                            continue
                        end_frame = num_frames
                    
                    # 存储子序列索引信息：(file_id, start_frame, end_frame)
                    self.sample_indices.append((file_id, start_frame, end_frame))
            
            else:  # 推理模式
                # 对于推理模式，我们只需要远端和麦克风信号
                features, _, _, num_frames, _ = prepare_features_for_inference(
                    farend_path, mic_path, self.seq_len,
                    self.target_sr, self.frame_length, self.hop_length, self.win_length
                )
                num_sequences = len(features)
                
                # 为每个子序列添加索引
                for seq_idx in range(num_sequences):
                    # 存储子序列索引信息：(file_id, seq_idx)
                    self.sample_indices.append((file_id, seq_idx))
                
                # 如果启用缓存，缓存特征
                if self.cache_features:
                    self.feature_cache[file_id] = features
        
        print(f"预处理完成，共生成 {len(self.sample_indices)} 个子序列样本")
    
    def __len__(self):
        """
        返回数据集中的样本数量
        
        返回:
            int: 样本数量（子序列数量）
        """
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: 
                训练模式: (far_end_features, mic_features, target_mask)
                推理模式: (far_end_features, mic_features)
                所有特征都是PyTorch张量
        """
        if self.mode == 'inference':
            # 推理模式
            file_id, seq_idx = self.sample_indices[idx]
            
            # 如果启用了缓存，从缓存中获取特征
            if self.cache_features and file_id in self.feature_cache:
                features = self.feature_cache[file_id][seq_idx]
                # 分离远端和麦克风特征
                freq_bins = features.shape[1] // 2
                mic_features = features[:, :freq_bins]
                far_end_features = features[:, freq_bins:]
                
                return torch.FloatTensor(far_end_features), torch.FloatTensor(mic_features)
            
            # 否则重新加载和处理
            farend_path = os.path.join(self.data_dir, f"{file_id}_farend.wav")
            mic_path = os.path.join(self.data_dir, f"{file_id}_mic.wav")
            
            features, _, _, _, _ = prepare_features_for_inference(
                farend_path, mic_path, self.seq_len,
                self.target_sr, self.frame_length, self.hop_length, self.win_length
            )
            
            # 获取指定的子序列
            features = features[seq_idx]
            
            # 分离远端和麦克风特征
            freq_bins = features.shape[1] // 2
            mic_features = features[:, :freq_bins]
            far_end_features = features[:, freq_bins:]
            
            return torch.FloatTensor(far_end_features), torch.FloatTensor(mic_features)
        
        else:  # 训练模式
            # 获取子序列索引信息
            file_id, start_frame, end_frame = self.sample_indices[idx]
            
            # 构建文件路径
            farend_path = os.path.join(self.data_dir, f"{file_id}_farend.wav")
            mic_path = os.path.join(self.data_dir, f"{file_id}_mic.wav")
            target_path = os.path.join(self.data_dir, f"{file_id}_target.wav")
            
            # 获取完整特征和掩码
            features, target_mask, _, _ = preprocess_audio(
                farend_path, mic_path, target_path,
                self.target_sr, self.frame_length, self.hop_length, self.win_length
            )
            
            # 提取子序列
            seq_features = features[start_frame:end_frame]
            seq_target_mask = target_mask[start_frame:end_frame]
            
            # 处理可能的填充（当子序列长度小于seq_len时）
            if seq_features.shape[0] < self.seq_len:
                pad_width = self.seq_len - seq_features.shape[0]
                seq_features = np.pad(seq_features, ((0, pad_width), (0, 0)), 'constant')
                seq_target_mask = np.pad(seq_target_mask, ((0, pad_width), (0, 0)), 'constant')
            
            # 分离远端和麦克风特征
            freq_bins = seq_features.shape[1] // 2
            mic_features = seq_features[:, :freq_bins]
            far_end_features = seq_features[:, freq_bins:]
            
            # 返回元组：(远端特征, 麦克风特征, 目标掩码)
            return (torch.FloatTensor(far_end_features), 
                    torch.FloatTensor(mic_features), 
                    torch.FloatTensor(seq_target_mask))

def create_dataloader(data_dir, meta_file, batch_size=32, seq_len=50, 
                      target_sr=16000, frame_length=320, hop_length=160, win_length=320,
                      mode='train', num_workers=4, shuffle=True, cache_features=False):
    """
    创建数据加载器
    
    参数:
        data_dir (str): 数据目录路径
        meta_file (str): 元数据CSV文件路径
        batch_size (int): 批次大小，表示每次训练使用的样本数量
        seq_len (int): 序列长度（帧数），设置为50帧
        target_sr (int): 目标采样率，设置为16000Hz
        frame_length (int): STFT帧长，设置为320点
        hop_length (int): STFT帧移，设置为160点
        win_length (int): STFT窗长，设置为320点
        mode (str): 'train'（训练模式）或'inference'（推理模式）
        num_workers (int): 数据加载的工作线程数，增加可提高数据加载速度
        shuffle (bool): 是否打乱数据，训练时通常设为True
        cache_features (bool): 是否缓存特征
        
    返回:
        DataLoader: PyTorch DataLoader对象，用于批量加载数据
    """
    # 创建数据集实例
    dataset = EchoCancellationDataset(
        data_dir=data_dir,
        meta_file=meta_file,
        seq_len=seq_len,
        target_sr=target_sr,
        frame_length=frame_length,
        hop_length=hop_length,
        win_length=win_length,
        mode=mode,
        cache_features=cache_features
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 将数据加载到固定内存中，加速GPU训练
    )
    
    return dataloader

def visualize_batch(far_end_features, mic_features, target_mask, sample_idx=0):
    """
    可视化批次中的一个样本
    
    参数:
        far_end_features (torch.Tensor): 远端特征，形状为 [batch_size, seq_len, freq_bins]
        mic_features (torch.Tensor): 麦克风特征，形状为 [batch_size, seq_len, freq_bins]
        target_mask (torch.Tensor): 目标掩码，形状为 [batch_size, seq_len, freq_bins]
        sample_idx (int): 要可视化的样本索引
    """
    # 转换为numpy数组
    far_end = far_end_features[sample_idx].numpy()
    mic = mic_features[sample_idx].numpy()
    mask = target_mask[sample_idx].numpy()
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 绘制远端特征
    im0 = axes[0].imshow(far_end.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('远端信号特征')
    axes[0].set_ylabel('频率')
    plt.colorbar(im0, ax=axes[0])
    
    # 绘制麦克风特征
    im1 = axes[1].imshow(mic.T, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('麦克风信号特征')
    axes[1].set_ylabel('频率')
    plt.colorbar(im1, ax=axes[1])
    
    # 绘制目标掩码
    im2 = axes[2].imshow(mask.T, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('理想掩码')
    axes[2].set_xlabel('时间帧')
    axes[2].set_ylabel('频率')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('batch_visualization.png')
    plt.close()
    print(f"可视化结果已保存为 'batch_visualization.png'")

def test_dataloader(args):
    """
    测试数据加载器
    
    参数:
        args: 命令行参数
    """
    print("\n" + "="*50)
    print("测试数据加载器")
    print("="*50)
    
    # 创建数据加载器
    print(f"\n创建{args.mode}数据加载器...")
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        meta_file=args.meta_file,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        target_sr=args.target_sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        mode=args.mode,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        cache_features=args.cache_features
    )
    
    # 测试数据加载速度
    print(f"\n测试数据加载速度（加载{min(5, len(dataloader))}个批次）...")
    start_time = time.time()
    
    # 加载几个批次并计算时间
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count >= 5:
            break
    
    elapsed_time = time.time() - start_time
    print(f"加载{batch_count}个批次耗时: {elapsed_time:.2f}秒")
    print(f"平均每批次加载时间: {elapsed_time/batch_count:.2f}秒")
    
    # 获取一个批次并检查形状
    print("\n获取一个批次并检查形状...")
    for batch in dataloader:
        if args.mode == 'train':
            far_end_features, mic_features, target_mask = batch
            print(f"远端特征形状: {far_end_features.shape}")
            print(f"麦克风特征形状: {mic_features.shape}")
            print(f"目标掩码形状: {target_mask.shape}")
            
            # 检查数值范围
            print("\n检查数值范围...")
            print(f"远端特征范围: [{far_end_features.min():.2f}, {far_end_features.max():.2f}]")
            print(f"麦克风特征范围: [{mic_features.min():.2f}, {mic_features.max():.2f}]")
            print(f"目标掩码范围: [{target_mask.min():.2f}, {target_mask.max():.2f}]")
            
            # 可视化批次
            if args.visualize:
                print("\n可视化批次...")
                visualize_batch(far_end_features, mic_features, target_mask)
        else:
            far_end_features, mic_features = batch
            print(f"远端特征形状: {far_end_features.shape}")
            print(f"麦克风特征形状: {mic_features.shape}")
            
            # 检查数值范围
            print("\n检查数值范围...")
            print(f"远端特征范围: [{far_end_features.min():.2f}, {far_end_features.max():.2f}]")
            print(f"麦克风特征范围: [{mic_features.min():.2f}, {mic_features.max():.2f}]")
        
        break  # 只检查一个批次
    
    print("\n数据加载器测试完成！")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试回声消除数据加载器")
    
    # 数据路径参数
    parser.add_argument('--data_dir', type=str, default='./data/synthetic', 
                        help='数据目录路径')
    parser.add_argument('--meta_file', type=str, default='./data/meta.csv', 
                        help='元数据CSV文件路径')
    
    # 数据处理参数
    parser.add_argument('--seq_len', type=int, default=50, 
                        help='序列长度（帧数）')
    parser.add_argument('--target_sr', type=int, default=16000, 
                        help='目标采样率')
    parser.add_argument('--frame_length', type=int, default=320, 
                        help='STFT帧长')
    parser.add_argument('--hop_length', type=int, default=160, 
                        help='STFT帧移')
    parser.add_argument('--win_length', type=int, default=320, 
                        help='STFT窗长')
    
    # 数据加载器参数
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批次大小')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='模式：train或inference')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载的工作线程数')
    parser.add_argument('--shuffle', action='store_true', 
                        help='是否打乱数据')
    parser.add_argument('--cache_features', action='store_true', 
                        help='是否缓存特征')
    parser.add_argument('--visualize', action='store_true', 
                        help='是否可视化批次')
    
    args = parser.parse_args()
    
    # 测试数据加载器
    test_dataloader(args)
