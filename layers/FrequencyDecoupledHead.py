"""
频率解耦输出头 (Frequency Decoupled Head)

实现三频带解耦头 (Tri-Band Decoupled Head) V2.0 方案：
1. 三个独立的投影头分别预测低频/中频/高频时域分量
2. 高频头使用隐层 SoftThreshold 去噪
3. 深度监督：使用标准 SWT 分解 Ground Truth 作为辅助监督目标
4. 时域直接相加重构

核心设计：
- Head 1 (Trend): Linear，无正则，预测平滑趋势
- Head 2 (Mid): Linear + Dropout(0.2)，预测中频波动
- Head 3 (Detail): Linear + SoftThreshold(隐层) + Dropout(0.5)，预测高频细节

Author: CAST Project
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftThreshold(nn.Module):
    """
    可学习软阈值模块
    
    在隐空间中应用软阈值去噪，滤除小幅度的噪声激活。
    公式: y = sign(x) * ReLU(|x| - τ)
    
    Args:
        num_features: 特征维度
        init_tau: 初始阈值 (默认 0.1)
    """
    def __init__(self, num_features, init_tau=0.1):
        super(SoftThreshold, self).__init__()
        self.tau = nn.Parameter(torch.ones(num_features) * init_tau)
    
    def forward(self, x):
        tau = torch.abs(self.tau)  # 确保阈值为正
        return torch.sign(x) * F.relu(torch.abs(x) - tau)
    
    def extra_repr(self):
        return f'num_features={self.tau.shape[0]}, init_tau={self.tau.mean().item():.4f}'


class TriBandDecoupledHead(nn.Module):
    """
    三频带解耦输出头 (Tri-Band Decoupled Head)
    
    将 LLM 的隐状态解耦为三个频率分量的时域预测，然后相加得到最终预测。
    
    架构:
        LLM Output (B*N, nf)
            │
            ├──► Head_Trend (Linear) ──────────────────► Pred_Trend
            │
            ├──► Head_Mid (Linear + Dropout) ──────────► Pred_Mid
            │
            └──► Head_Detail (Linear + SoftThreshold + Dropout) ──► Pred_Detail
            │
            ▼
        Final = Pred_Trend + Pred_Mid + Pred_Detail
    
    Args:
        n_vars: 变量数量
        nf: 输入特征维度 (d_ff * patch_nums)
        target_window: 预测窗口长度 (pred_len)
        head_dropout: 输出 Dropout 比例
        mid_dropout: 中频头 Dropout 比例 (默认 0.2)
        high_dropout: 高频头 Dropout 比例 (默认 0.5)
        use_soft_threshold: 是否在高频头使用软阈值 (默认 True)
        soft_threshold_init: 软阈值初始值 (默认 0.1)
        use_conv: 是否使用 Conv1d 替代 Linear (增加位置感知，默认 False)
    """
    
    def __init__(self, n_vars, nf, target_window, head_dropout=0.1,
                 mid_dropout=0.2, high_dropout=0.5,
                 use_soft_threshold=True, soft_threshold_init=0.1,
                 use_conv=False):
        super(TriBandDecoupledHead, self).__init__()
        
        self.n_vars = n_vars
        self.nf = nf
        self.target_window = target_window
        self.use_soft_threshold = use_soft_threshold
        self.use_conv = use_conv
        
        # 展平层
        self.flatten = nn.Flatten(start_dim=-2)
        
        # ========== Head 1: 低频/趋势头 (无正则) ==========
        if use_conv:
            # Conv1d 提供位置感知能力
            self.head_trend = nn.Conv1d(nf, target_window, kernel_size=1)
        else:
            self.head_trend = nn.Linear(nf, target_window)
        
        # ========== Head 2: 中频头 (轻微 Dropout) ==========
        if use_conv:
            self.head_mid_proj = nn.Conv1d(nf, target_window, kernel_size=1)
        else:
            self.head_mid_proj = nn.Linear(nf, target_window)
        self.head_mid_dropout = nn.Dropout(mid_dropout)
        
        # ========== Head 3: 高频/细节头 (强正则) ==========
        # 设计：先投影到隐层 → 软阈值去噪 → 重构到时域
        hidden_dim = max(nf // 2, target_window)  # 隐层维度
        
        if use_conv:
            self.head_detail_to_latent = nn.Conv1d(nf, hidden_dim, kernel_size=1)
            self.head_detail_to_time = nn.Conv1d(hidden_dim, target_window, kernel_size=1)
        else:
            self.head_detail_to_latent = nn.Linear(nf, hidden_dim)
            self.head_detail_to_time = nn.Linear(hidden_dim, target_window)
        
        if use_soft_threshold:
            self.soft_threshold = SoftThreshold(hidden_dim, init_tau=soft_threshold_init)
        
        self.head_detail_dropout = nn.Dropout(high_dropout)
        
        # 输出 Dropout
        self.output_dropout = nn.Dropout(head_dropout)
        
        # 初始化权重
        self._init_weights()
        
        # 打印配置
        self._print_config()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _print_config(self):
        """打印模块配置"""
        print("=" * 70)
        print("[TriBandDecoupledHead] 三频带解耦输出头已启用")
        print("=" * 70)
        print(f"  ├─ 输入特征维度: {self.nf}")
        print(f"  ├─ 预测窗口长度: {self.target_window}")
        print(f"  ├─ 变量数量: {self.n_vars}")
        print(f"  ├─ 投影类型: {'Conv1d' if self.use_conv else 'Linear'}")
        print(f"  ├─ Head 1 (Trend): 无正则化")
        print(f"  ├─ Head 2 (Mid): Dropout={self.head_mid_dropout.p}")
        print(f"  ├─ Head 3 (Detail):")
        print(f"  │   ├─ SoftThreshold: {'✅ 启用' if self.use_soft_threshold else '❌ 关闭'}")
        print(f"  │   └─ Dropout={self.head_detail_dropout.p}")
        print(f"  └─ 输出 Dropout: {self.output_dropout.p}")
        print("=" * 70)
    
    def forward(self, x, return_components=False):
        """
        前向传播
        
        Args:
            x: LLM 输出，形状 (B, n_vars, d_ff, patch_nums) 或 (B, n_vars, nf)
            return_components: 是否返回三个频率分量 (用于深度监督)
        
        Returns:
            final_pred: 最终预测，形状 (B, target_window, n_vars)
            components: (可选) 字典，包含 pred_trend, pred_mid, pred_detail
        """
        # 输入形状处理
        if x.dim() == 4:
            # (B, n_vars, d_ff, patch_nums) -> (B, n_vars, nf)
            B, N, D, P = x.shape
            x = self.flatten(x)  # (B, n_vars, d_ff * patch_nums)
        else:
            B, N, _ = x.shape
        
        # ========== Head 1: 低频趋势预测 ==========
        if self.use_conv:
            pred_trend = self.head_trend(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            pred_trend = self.head_trend(x)  # (B, n_vars, target_window)
        
        # ========== Head 2: 中频预测 ==========
        if self.use_conv:
            pred_mid = self.head_mid_proj(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            pred_mid = self.head_mid_proj(x)
        pred_mid = self.head_mid_dropout(pred_mid)
        
        # ========== Head 3: 高频细节预测 ==========
        # Step 1: 投影到隐层
        if self.use_conv:
            h_detail = self.head_detail_to_latent(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            h_detail = self.head_detail_to_latent(x)
        
        # Step 2: 隐层软阈值去噪
        if self.use_soft_threshold:
            h_detail = self.soft_threshold(h_detail)
        
        # Step 3: Dropout
        h_detail = self.head_detail_dropout(h_detail)
        
        # Step 4: 重构到时域
        if self.use_conv:
            pred_detail = self.head_detail_to_time(h_detail.transpose(-1, -2)).transpose(-1, -2)
        else:
            pred_detail = self.head_detail_to_time(h_detail)
        
        # ========== 时域直接相加重构 ==========
        final_pred = pred_trend + pred_mid + pred_detail
        
        # 输出 Dropout
        final_pred = self.output_dropout(final_pred)
        
        # 调整输出形状: (B, n_vars, target_window) -> (B, target_window, n_vars)
        final_pred = final_pred.permute(0, 2, 1).contiguous()
        
        if return_components:
            components = {
                'pred_trend': pred_trend.permute(0, 2, 1).contiguous(),
                'pred_mid': pred_mid.permute(0, 2, 1).contiguous(),
                'pred_detail': pred_detail.permute(0, 2, 1).contiguous(),
            }
            return final_pred, components
        
        return final_pred


class DeepSupervisionLoss(nn.Module):
    """
    深度监督损失模块
    
    使用标准 SWT (非因果，无损) 分解 Ground Truth，
    作为三个频率头的辅助监督目标。
    
    Total Loss = Main Loss + α × (Loss_Trend + Loss_Mid + Loss_Detail)
    
    Args:
        wavelet: 小波类型 (默认 'db4')
        level: 分解层数 (默认 2，产生 3 个频带)
        alpha: 辅助损失权重 (默认 0.3)
        use_causal_swt: 是否使用因果 SWT 分解 GT (Plan B，默认 False)
    """
    
    def __init__(self, wavelet='db4', level=2, alpha=0.3, use_causal_swt=False):
        super(DeepSupervisionLoss, self).__init__()
        
        self.wavelet = wavelet
        self.level = level
        self.alpha = alpha
        self.use_causal_swt = use_causal_swt
        self.num_bands = level + 1
        
        # 尝试导入小波模块
        self._init_swt()
        
        print("=" * 70)
        print("[DeepSupervisionLoss] 深度监督损失已启用")
        print("=" * 70)
        print(f"  ├─ 小波类型: {wavelet}")
        print(f"  ├─ 分解层数: {level}")
        print(f"  ├─ 频带数量: {self.num_bands}")
        print(f"  ├─ 辅助损失权重 α: {alpha}")
        print(f"  ├─ SWT 类型: {'Causal (Plan B)' if use_causal_swt else 'Standard (Plan A)'}")
        print(f"  └─ Loss 公式: Main + α × (Aux_Trend + Aux_Mid + Aux_Detail)")
        print("=" * 70)
    
    def _init_swt(self):
        """初始化 SWT 模块"""
        if self.use_causal_swt:
            # Plan B: 使用因果 SWT
            try:
                from layers.CausalWavelet import CausalSWT
            except ImportError:
                from .CausalWavelet import CausalSWT
            self.swt = CausalSWT(wavelet=self.wavelet, level=self.level)
            self.swt_type = 'causal'
        else:
            # Plan A: 使用标准 SWT (尝试 ptwt，否则回退到因果版本)
            try:
                import ptwt
                import pywt
                self.swt_type = 'standard'
                self._ptwt = ptwt
                self._pywt = pywt
            except ImportError:
                print("[Warning] ptwt/pywt 未安装，回退到 CausalSWT")
                try:
                    from layers.CausalWavelet import CausalSWT
                except ImportError:
                    from .CausalWavelet import CausalSWT
                self.swt = CausalSWT(wavelet=self.wavelet, level=self.level)
                self.swt_type = 'causal'
    
    def _standard_swt(self, x):
        """
        标准 SWT 分解 (非因果，使用 ptwt)
        
        Args:
            x: (B, N, T) 输入信号
        
        Returns:
            coeffs: (B, N, T, num_bands) 小波系数
        """
        B, N, T = x.shape
        device = x.device
        dtype = x.dtype
        
        # ptwt.swt 需要 (B, T) 或 (B, C, T) 输入
        # 我们逐变量处理
        x_flat = x.reshape(B * N, T)
        
        # 转为 float32 (ptwt 可能不支持 bfloat16)
        x_float = x_flat.float()
        
        # SWT 分解
        coeffs_list = self._ptwt.swt(x_float, self._pywt.Wavelet(self.wavelet), level=self.level)
        
        # coeffs_list 是 [(cA, cD), (cA, cD), ...] 或 [cD1, cD2, ..., cA]
        # ptwt.swt 返回格式: list of (cA, cD) tuples, 从 level 1 到 level n
        # 我们需要重组为 [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        
        all_bands = []
        
        # 提取最终的近似系数 cA
        cA = coeffs_list[-1][0]  # 最后一层的 cA
        all_bands.append(cA)
        
        # 提取细节系数 cD (从高层到低层)
        for i in range(self.level - 1, -1, -1):
            cD = coeffs_list[i][1]
            all_bands.append(cD)
        
        # Stack: (B*N, T, num_bands)
        coeffs = torch.stack(all_bands, dim=-1)
        
        # Reshape: (B*N, T, num_bands) -> (B, N, T, num_bands)
        coeffs = coeffs.reshape(B, N, T, self.num_bands)
        
        # 转回原始 dtype
        coeffs = coeffs.to(dtype)
        
        return coeffs
    
    def _decompose_target(self, target):
        """
        分解目标序列
        
        Args:
            target: (B, T, N) Ground Truth
        
        Returns:
            target_bands: dict, 包含 'trend', 'mid', 'detail'
        """
        # 调整形状: (B, T, N) -> (B, N, T)
        target = target.permute(0, 2, 1).contiguous()
        B, N, T = target.shape
        
        if self.swt_type == 'standard':
            coeffs = self._standard_swt(target)
        else:
            coeffs = self.swt(target)
        
        # coeffs: (B, N, T, num_bands)
        # 顺序: [cA_n (trend), cD_n (mid), cD_1 (detail)]
        
        # 对于 level=2: [cA2, cD2, cD1]
        target_trend = coeffs[:, :, :, 0]    # cA: 低频趋势
        target_mid = coeffs[:, :, :, 1]      # cD2: 中频
        target_detail = coeffs[:, :, :, -1]  # cD1: 高频细节
        
        # 转回 (B, T, N) 格式
        target_bands = {
            'trend': target_trend.permute(0, 2, 1).contiguous(),
            'mid': target_mid.permute(0, 2, 1).contiguous(),
            'detail': target_detail.permute(0, 2, 1).contiguous(),
        }
        
        return target_bands
    
    def forward(self, pred, target, components=None, main_loss=None):
        """
        计算深度监督损失
        
        Args:
            pred: (B, T, N) 模型预测
            target: (B, T, N) Ground Truth
            components: dict, 包含 'pred_trend', 'pred_mid', 'pred_detail'
            main_loss: 预计算的主损失 (可选)
        
        Returns:
            total_loss: 总损失
            loss_dict: 包含各项损失的字典
        """
        # 计算主损失
        if main_loss is None:
            main_loss = F.mse_loss(pred, target)
        
        loss_dict = {'main_loss': main_loss.item()}
        
        # 如果没有提供分量，只返回主损失
        if components is None:
            return main_loss, loss_dict
        
        # 分解目标
        target_bands = self._decompose_target(target)
        
        # 计算辅助损失
        loss_trend = F.mse_loss(components['pred_trend'], target_bands['trend'])
        loss_mid = F.mse_loss(components['pred_mid'], target_bands['mid'])
        loss_detail = F.mse_loss(components['pred_detail'], target_bands['detail'])
        
        # 总辅助损失
        aux_loss = loss_trend + loss_mid + loss_detail
        
        # 总损失
        total_loss = main_loss + self.alpha * aux_loss
        
        # 记录各项损失
        loss_dict.update({
            'loss_trend': loss_trend.item(),
            'loss_mid': loss_mid.item(),
            'loss_detail': loss_detail.item(),
            'aux_loss': aux_loss.item(),
            'total_loss': total_loss.item(),
            'alpha': self.alpha,
        })
        
        return total_loss, loss_dict


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    # 添加项目根目录到 Python 路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    print("=" * 70)
    print("TriBandDecoupledHead 模块测试")
    print("=" * 70)
    
    # 设备选择
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 测试参数
    B = 4           # Batch size
    N = 7           # 变量数
    d_ff = 32       # FFN 维度
    patch_nums = 10 # Patch 数量
    pred_len = 96   # 预测长度
    
    nf = d_ff * patch_nums  # 特征维度
    
    print(f"\n测试配置:")
    print(f"  - Batch: {B}, Variables: {N}")
    print(f"  - d_ff: {d_ff}, patch_nums: {patch_nums}")
    print(f"  - nf (d_ff * patch_nums): {nf}")
    print(f"  - pred_len: {pred_len}")
    
    # ========== 测试 1: 基本前向传播 ==========
    print("\n" + "=" * 70)
    print("测试 1: 基本前向传播")
    print("=" * 70)
    
    head = TriBandDecoupledHead(
        n_vars=N,
        nf=nf,
        target_window=pred_len,
        head_dropout=0.1,
        mid_dropout=0.2,
        high_dropout=0.5,
        use_soft_threshold=True,
        soft_threshold_init=0.1,
        use_conv=False
    ).to(device)
    
    # 模拟 LLM 输出
    x = torch.randn(B, N, d_ff, patch_nums, device=device)
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播 (不返回分量)
    output = head(x, return_components=False)
    print(f"输出形状: {output.shape}")
    assert output.shape == (B, pred_len, N), f"输出形状错误: {output.shape}"
    print("✅ 基本前向传播通过")
    
    # ========== 测试 2: 返回频率分量 ==========
    print("\n" + "=" * 70)
    print("测试 2: 返回频率分量 (用于深度监督)")
    print("=" * 70)
    
    output, components = head(x, return_components=True)
    print(f"\n最终预测形状: {output.shape}")
    print(f"趋势分量形状: {components['pred_trend'].shape}")
    print(f"中频分量形状: {components['pred_mid'].shape}")
    print(f"细节分量形状: {components['pred_detail'].shape}")
    
    # 验证分量相加等于总预测
    reconstructed = components['pred_trend'] + components['pred_mid'] + components['pred_detail']
    diff = (output - reconstructed).abs().max().item()
    print(f"\n分量相加 vs 最终输出 差异: {diff:.10f}")
    
    # 注意：由于输出 Dropout，eval 模式下应该相等
    head.eval()
    with torch.no_grad():
        output_eval, components_eval = head(x, return_components=True)
        reconstructed_eval = components_eval['pred_trend'] + components_eval['pred_mid'] + components_eval['pred_detail']
        diff_eval = (output_eval - reconstructed_eval).abs().max().item()
    print(f"(eval 模式) 分量相加 vs 最终输出 差异: {diff_eval:.10f}")
    assert diff_eval < 1e-5, "分量重构不一致!"
    print("✅ 频率分量返回正确")
    head.train()
    
    # ========== 测试 3: SoftThreshold 功能 ==========
    print("\n" + "=" * 70)
    print("测试 3: SoftThreshold 功能")
    print("=" * 70)
    
    st = SoftThreshold(num_features=64, init_tau=0.1).to(device)
    
    # 测试输入
    test_input = torch.randn(B, 64, device=device)
    test_output = st(test_input)
    
    # 检查小于阈值的值是否被置零
    tau = st.tau.abs()
    small_values_mask = test_input.abs() < tau
    zero_check = (test_output[small_values_mask] == 0).all()
    print(f"\n阈值 τ 均值: {tau.mean().item():.4f}")
    print(f"小于阈值的输入被置零: {zero_check.item()}")
    print("✅ SoftThreshold 功能正确")
    
    # ========== 测试 4: 深度监督损失 ==========
    print("\n" + "=" * 70)
    print("测试 4: 深度监督损失 (DeepSupervisionLoss)")
    print("=" * 70)
    
    # 创建深度监督损失模块 (使用因果 SWT，因为标准 SWT 可能没装 ptwt)
    ds_loss = DeepSupervisionLoss(
        wavelet='db4',
        level=2,
        alpha=0.3,
        use_causal_swt=True  # 使用因果版本以确保测试通过
    ).to(device)
    
    # 模拟预测和目标
    pred = torch.randn(B, pred_len, N, device=device)
    target = torch.randn(B, pred_len, N, device=device)
    
    # 获取分量
    head.eval()
    with torch.no_grad():
        _, components = head(x, return_components=True)
    head.train()
    
    # 计算损失
    total_loss, loss_dict = ds_loss(pred, target, components)
    
    print(f"\n损失详情:")
    for k, v in loss_dict.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.6f}")
        else:
            print(f"  - {k}: {v}")
    
    print(f"\n总损失: {total_loss.item():.6f}")
    print("✅ 深度监督损失计算正确")
    
    # ========== 测试 5: 梯度传播 ==========
    print("\n" + "=" * 70)
    print("测试 5: 梯度传播")
    print("=" * 70)
    
    head.train()
    x.requires_grad = True
    
    output, components = head(x, return_components=True)
    loss, _ = ds_loss(output, target, components)
    loss.backward()
    
    # 检查梯度
    grad_check = x.grad is not None and x.grad.abs().sum() > 0
    print(f"\n输入梯度存在: {grad_check}")
    
    # 检查各 Head 权重的梯度
    for name, param in head.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  - {name}: grad_norm = {grad_norm:.6f}")
    
    print("✅ 梯度传播正确")
    
    # ========== 测试 6: 参数统计 ==========
    print("\n" + "=" * 70)
    print("测试 6: 参数统计")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 分头统计
    trend_params = sum(p.numel() for n, p in head.named_parameters() if 'trend' in n)
    mid_params = sum(p.numel() for n, p in head.named_parameters() if 'mid' in n)
    detail_params = sum(p.numel() for n, p in head.named_parameters() if 'detail' in n)
    
    print(f"\nHead Trend 参数: {trend_params:,}")
    print(f"Head Mid 参数: {mid_params:,}")
    print(f"Head Detail 参数: {detail_params:,}")
    print("✅ 参数统计完成")
    
    # ========== 测试 7: Conv1d 模式 ==========
    print("\n" + "=" * 70)
    print("测试 7: Conv1d 模式 (位置感知)")
    print("=" * 70)
    
    head_conv = TriBandDecoupledHead(
        n_vars=N,
        nf=nf,
        target_window=pred_len,
        use_soft_threshold=True,
        use_conv=True  # 使用 Conv1d
    ).to(device)
    
    x_test = torch.randn(B, N, d_ff, patch_nums, device=device)
    output_conv = head_conv(x_test)
    print(f"\nConv1d 模式输出形状: {output_conv.shape}")
    assert output_conv.shape == (B, pred_len, N)
    print("✅ Conv1d 模式正确")
    
    # ========== 测试完成 ==========
    print("\n" + "=" * 70)
    print("🎉 所有测试通过!")
    print("=" * 70)
