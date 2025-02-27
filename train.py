import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from preprocess import CachedAECDataset, Config
from model.transfomer import TransformerEchoCancellation


def parse_args():
    parser = argparse.ArgumentParser(description='训练回声消除Transformer模型')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')  # 默认减小批次大小
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--save_interval', type=int, default=5, help='模型保存间隔（轮数）')
    parser.add_argument('--log_interval', type=int, default=10, help='日志打印间隔（批次数）')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--use_cuda', action='store_true', help='使用CUDA')
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--mixed_precision', action='store_true', help='使用混合精度训练')
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


def load_datasets(config, batch_size):
    # 加载训练集和验证集
    train_dataset = CachedAECDataset(
        os.path.join(config.output_dir, 'train.h5'),
        max_seq_len=config.max_seq_len,
        stride=config.max_seq_len // 2  # 50%重叠
    )

    val_dataset = CachedAECDataset(
        os.path.join(config.output_dir, 'val.h5'),
        max_seq_len=config.max_seq_len,
        stride=config.max_seq_len  # 验证集不需要重叠
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    # 创建检查点目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存模型状态
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)

    print(f"检查点已保存至 {save_path}")


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_interval, grad_accum_steps=1,
                scaler=None):
    model.train()
    total_loss = 0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as pbar:
        for batch_idx, (features, targets) in enumerate(pbar):
            # 将数据移至GPU
            features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 确定是否需要累积梯度
            is_accumulation_step = ((batch_idx + 1) % grad_accum_steps != 0)

            # 混合精度训练
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # 前向传播
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    # 根据梯度累积步数缩放损失
                    loss = loss / grad_accum_steps

                # 反向传播
                scaler.scale(loss).backward()

                if not is_accumulation_step:
                    # 梯度裁剪，防止梯度爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # 优化器步进
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            else:
                # 常规训练流程
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, targets)
                # 根据梯度累积步数缩放损失
                loss = loss / grad_accum_steps

                # 反向传播
                loss.backward()

                if not is_accumulation_step:
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # 优化器步进
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            # 累计损失 (使用未缩放的损失值进行显示)
            batch_loss = loss.item() * grad_accum_steps
            total_loss += batch_loss

            # 更新进度条
            pbar.set_postfix(loss=batch_loss)

            # 打印日志
            if (batch_idx + 1) % log_interval == 0:
                print(f'Train Epoch: {epoch + 1} [{batch_idx + 1}/{len(train_loader)}] Loss: {batch_loss:.6f}')

            # 定期清理缓存以防止内存泄漏
            if torch.cuda.is_available() and (batch_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1} 训练平均损失: {avg_loss:.6f}')

    return avg_loss


def validate(model, val_loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, targets in tqdm(val_loader, desc="Validating", unit="batch"):
            # 将数据移至GPU
            features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 前向传播 (使用混合精度，如果启用)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(features)
                loss = criterion(outputs, targets)

            # 累计损失
            total_loss += loss.item()

            # 在使用大批量或大模型时，定期清理缓存
            if torch.cuda.is_available() and (val_loader.batch_size >= 16):
                torch.cuda.empty_cache()

    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    print(f'验证平均损失: {avg_loss:.6f}')

    return avg_loss


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置设备
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")

    # 设置CUDA内存优化
    if use_cuda:
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        # 设置较为保守的内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.8)  # 只使用80%的GPU内存
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"当前分配的内存: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

    # 加载配置
    config = Config()
    config.batch_size = args.batch_size

    # 创建模型
    model = create_model(config)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 使用二元交叉熵损失函数，适用于0-1之间的掩码预测
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 设置混合精度训练
    scaler = None
    if args.mixed_precision and use_cuda:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("启用混合精度训练")

    # 加载数据集
    train_loader, val_loader = load_datasets(config, args.batch_size)

    # 用于存储损失历史
    train_losses = []
    val_losses = []

    # 创建保存模型的目录
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 恢复训练（如果指定）
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"加载检查点: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"从第 {start_epoch} 轮开始训练")
        else:
            print(f"检查点 {args.resume} 不存在，从头开始训练")

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # 训练前清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 训练一轮
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.log_interval,
            args.gradient_accumulation,
            scaler
        )
        train_losses.append(train_loss)

        # 训练后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 验证
        val_loss = validate(model, val_loader, criterion, device, scaler)
        val_losses.append(val_loss)
        
        # 更新学习率调度器
        scheduler.step(val_loss)

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # 绘制损失曲线
        plot_path = "loss_curve.png"
        plot_losses(train_losses, val_losses, plot_path)

        # 打印轮次时间和内存使用情况
        epoch_time = time.time() - start_time
        print(f"第 {epoch + 1} 轮用时: {epoch_time:.2f} 秒")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        if torch.cuda.is_available():
            print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB / "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pt")
    save_checkpoint(model, optimizer, args.epochs - 1, val_losses[-1], final_checkpoint_path)

    print("训练完成！")


if __name__ == "__main__":
    main()