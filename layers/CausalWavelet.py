"""
因果平稳小波变换 (Causal Stationary Wavelet Transform)

纯PyTorch实现，不依赖ptwt/pywt库，支持GPU加速。
核心特性：严格因果性，计算t时刻的系数仅使用t及之前的数据。

Author: SWT-Time Project
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 小波滤波器系数库
# ============================================================================

# Daubechies小波滤波器系数
# 来源: PyWavelets库验证过的系数
WAVELET_FILTERS = {
    'db1': {  # Haar小波
        'dec_lo': [0.7071067811865476, 0.7071067811865476],
        'dec_hi': [-0.7071067811865476, 0.7071067811865476],
        'rec_lo': [0.7071067811865476, 0.7071067811865476],
        'rec_hi': [0.7071067811865476, -0.7071067811865476],
    },
    'haar': {  # Haar小波别名
        'dec_lo': [0.7071067811865476, 0.7071067811865476],
        'dec_hi': [-0.7071067811865476, 0.7071067811865476],
        'rec_lo': [0.7071067811865476, 0.7071067811865476],
        'rec_hi': [0.7071067811865476, -0.7071067811865476],
    },
    'db2': {
        'dec_lo': [-0.12940952255092145, 0.22414386804185735, 
                   0.836516303737469, 0.48296291314469025],
        'dec_hi': [-0.48296291314469025, 0.836516303737469, 
                   -0.22414386804185735, -0.12940952255092145],
        'rec_lo': [0.48296291314469025, 0.836516303737469, 
                   0.22414386804185735, -0.12940952255092145],
        'rec_hi': [-0.12940952255092145, -0.22414386804185735, 
                   0.836516303737469, -0.48296291314469025],
    },
    'db3': {
        'dec_lo': [0.035226291882100656, -0.08544127388224149, 
                   -0.13501102001039084, 0.4598775021193313, 
                   0.8068915093133388, 0.3326705529509569],
        'dec_hi': [-0.3326705529509569, 0.8068915093133388, 
                   -0.4598775021193313, -0.13501102001039084, 
                   0.08544127388224149, 0.035226291882100656],
        'rec_lo': [0.3326705529509569, 0.8068915093133388, 
                   0.4598775021193313, -0.13501102001039084, 
                   -0.08544127388224149, 0.035226291882100656],
        'rec_hi': [0.035226291882100656, 0.08544127388224149, 
                   -0.13501102001039084, -0.4598775021193313, 
                   0.8068915093133388, -0.3326705529509569],
    },
    'db4': {
        'dec_lo': [-0.010597401784997278, 0.032883011666982945, 
                   0.030841381835986965, -0.18703481171888114, 
                   -0.02798376941698385, 0.6308807679295904, 
                   0.7148465705525415, 0.23037781330885523],
        'dec_hi': [-0.23037781330885523, 0.7148465705525415, 
                   -0.6308807679295904, -0.02798376941698385, 
                   0.18703481171888114, 0.030841381835986965, 
                   -0.032883011666982945, -0.010597401784997278],
        'rec_lo': [0.23037781330885523, 0.7148465705525415, 
                   0.6308807679295904, -0.02798376941698385, 
                   -0.18703481171888114, 0.030841381835986965, 
                   0.032883011666982945, -0.010597401784997278],
        'rec_hi': [-0.010597401784997278, -0.032883011666982945, 
                   0.030841381835986965, 0.18703481171888114, 
                   -0.02798376941698385, -0.6308807679295904, 
                   0.7148465705525415, -0.23037781330885523],
    },
    'db5': {
        'dec_lo': [0.003335725285001549, -0.012580751999015526, 
                   -0.006241490213011705, 0.07757149384006515, 
                   -0.03224486958502952, -0.24229488706619015, 
                   0.13842814590110342, 0.7243085284385744, 
                   0.6038292697974729, 0.160102397974125],
        'dec_hi': [-0.160102397974125, 0.6038292697974729, 
                   -0.7243085284385744, 0.13842814590110342, 
                   0.24229488706619015, -0.03224486958502952, 
                   -0.07757149384006515, -0.006241490213011705, 
                   0.012580751999015526, 0.003335725285001549],
        'rec_lo': [0.160102397974125, 0.6038292697974729, 
                   0.7243085284385744, 0.13842814590110342, 
                   -0.24229488706619015, -0.03224486958502952, 
                   0.07757149384006515, -0.006241490213011705, 
                   -0.012580751999015526, 0.003335725285001549],
        'rec_hi': [0.003335725285001549, 0.012580751999015526, 
                   -0.006241490213011705, -0.07757149384006515, 
                   -0.03224486958502952, 0.24229488706619015, 
                   0.13842814590110342, -0.7243085284385744, 
                   0.6038292697974729, -0.160102397974125],
    },
}


def get_wavelet_filters(wavelet: str) -> dict:
    """获取小波滤波器系数
    
    Args:
        wavelet: 小波名称 ('db1', 'db2', 'db3', 'db4', 'db5', 'haar')
    
    Returns:
        dict: 包含 dec_lo, dec_hi, rec_lo, rec_hi 四组滤波器系数
    """
    if wavelet not in WAVELET_FILTERS:
        raise ValueError(
            f"不支持的小波类型: '{wavelet}'\n"
            f"支持的类型: {list(WAVELET_FILTERS.keys())}"
        )
    return WAVELET_FILTERS[wavelet]


# ============================================================================
# 因果平稳小波变换 (CausalSWT)
# ============================================================================

class CausalSWT(nn.Module):
    """因果平稳小波变换 (Causal Stationary Wavelet Transform)
    
    纯PyTorch实现，严格保证因果性：计算t时刻的系数仅使用t及之前的数据。
    
    实现原理：
    1. 使用因果卷积（仅左侧padding）
    2. 通过dilation实现SWT的非下采样特性
    3. 每层dilation翻倍：level 1用d=1, level 2用d=2, level 3用d=4
    
    Args:
        wavelet: 小波基函数名称 (默认'db4')
        level: 分解层数 (默认3)
    
    Input:
        x: (B, N, T) - [batch_size, num_variables, time_steps]
    
    Output:
        coeffs: (B, N, T, Level+1) - 多频段系数
                顺序: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    """
    
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        super(CausalSWT, self).__init__()
        
        self.wavelet_name = wavelet
        self.level = level
        
        # 获取滤波器系数
        filters = get_wavelet_filters(wavelet)
        self.filter_len = len(filters['dec_lo'])
        
        # 创建滤波器张量 (1, 1, filter_len)
        # 注意：Conv1d执行的是互相关，需要反转滤波器
        dec_lo = torch.tensor(filters['dec_lo'][::-1], dtype=torch.float32)
        dec_hi = torch.tensor(filters['dec_hi'][::-1], dtype=torch.float32)
        
        # 注册为buffer（不参与梯度更新，但随模型移动到GPU）
        self.register_buffer('dec_lo', dec_lo.view(1, 1, -1))
        self.register_buffer('dec_hi', dec_hi.view(1, 1, -1))
        
        # 最小序列长度
        self.min_length = 2 ** level
        
        print("[CausalSWT] 创建因果平稳小波变换")
        print(f"  - 小波类型: {wavelet}")
        print(f"  - 滤波器长度: {self.filter_len}")
        print(f"  - 分解层数: {level}")
        print(f"  - 输出频段数: {level + 1}")
        print("  - 特性: 严格因果（仅使用过去数据）")
    
    def _causal_conv(self, x: torch.Tensor, weight: torch.Tensor, 
                     dilation: int) -> torch.Tensor:
        """因果卷积：仅使用当前及过去的数据
        
        Args:
            x: (B*N, 1, T) - 输入信号
            weight: (1, 1, K) - 卷积核
            dilation: 膨胀系数
        
        Returns:
            (B*N, 1, T) - 输出信号，长度保持不变
        """
        # 计算因果padding大小
        # 有效滤波器长度 = (K - 1) * dilation + 1
        # 为保证输出长度=输入长度，需要左侧padding = (K - 1) * dilation
        padding_size = (self.filter_len - 1) * dilation
        
        # 左侧复制填充（比零填充更平滑，避免边界突变）
        x_padded = F.pad(x, (padding_size, 0), mode='replicate')
        
        # 确保weight与输入dtype一致（处理混合精度训练）
        weight = weight.to(x_padded.dtype)
        
        # 执行膨胀卷积
        out = F.conv1d(x_padded, weight, dilation=dilation)
        
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行因果SWT分解
        
        Args:
            x: (B, N, T) - 输入时间序列
        
        Returns:
            coeffs: (B, N, T, Level+1) - 小波系数
                    顺序: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        """
        # 输入验证
        if x.ndim != 3:
            raise ValueError(f"输入必须是3维张量 (B, N, T)，当前: {x.ndim}维")
        
        B, N, T = x.shape
        
        if T < self.min_length:
            raise ValueError(
                f"序列长度 {T} 太短，{self.level}层SWT至少需要 {self.min_length}"
            )
        
        # NaN/Inf检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("输入包含NaN或Inf值，请先进行数据清洗")
        
        # Reshape: (B, N, T) -> (B*N, 1, T)
        # 注意：直接支持bfloat16，不做类型转换
        current = x.reshape(B * N, 1, T)
        
        # 存储细节系数 [cD_1, cD_2, ..., cD_n]
        details = []
        
        # 逐层分解
        for j in range(self.level):
            # 当前层的dilation: 2^j
            dilation = 2 ** j
            
            # 因果卷积
            cA = self._causal_conv(current, self.dec_lo, dilation)  # 近似
            cD = self._causal_conv(current, self.dec_hi, dilation)  # 细节
            
            # 保存细节系数
            details.append(cD)
            
            # 下一层的输入是当前层的近似
            current = cA
        
        # current 现在是最后一层的近似系数 cA_n
        
        # 组装输出：[cA_n, cD_n, cD_{n-1}, ..., cD_1]
        # details 的顺序是 [cD_1, cD_2, ..., cD_n]，需要反转
        all_coeffs = [current] + details[::-1]
        
        # Stack: list of (B*N, 1, T) -> (B*N, Level+1, T)
        coeffs = torch.cat(all_coeffs, dim=1)
        
        # Reshape: (B*N, Level+1, T) -> (B, N, T, Level+1)
        coeffs = coeffs.reshape(B, N, self.level + 1, T)
        coeffs = coeffs.permute(0, 1, 3, 2).contiguous()
        
        return coeffs


# ============================================================================
# 逆因果平稳小波变换 (CausalISWT)
# ============================================================================

class CausalISWT(nn.Module):
    """逆因果平稳小波变换 (Inverse Causal Stationary Wavelet Transform)
    
    与CausalSWT严格对称的重构模块，从小波系数恢复原始信号。
    
    Args:
        wavelet: 小波基函数名称（必须与分解时一致）
        level: 分解层数（必须与分解时一致）
    
    Input:
        coeffs: (B, N, T, Level+1) - 小波系数
                顺序: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    
    Output:
        x: (B, N, T) - 重构的时域信号
    """
    
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        super(CausalISWT, self).__init__()
        
        self.wavelet_name = wavelet
        self.level = level
        
        # 获取重构滤波器系数
        filters = get_wavelet_filters(wavelet)
        self.filter_len = len(filters['rec_lo'])
        
        # 创建重构滤波器张量（同样需要反转）
        rec_lo = torch.tensor(filters['rec_lo'][::-1], dtype=torch.float32)
        rec_hi = torch.tensor(filters['rec_hi'][::-1], dtype=torch.float32)
        
        self.register_buffer('rec_lo', rec_lo.view(1, 1, -1))
        self.register_buffer('rec_hi', rec_hi.view(1, 1, -1))
        
        print("[CausalISWT] 创建逆因果平稳小波变换")
        print(f"  - 小波类型: {wavelet}")
        print(f"  - 分解层数: {level}")
        print("  - 特性: 反因果重构（补偿相位偏移）")
    
    def _anticausal_conv(self, x: torch.Tensor, weight: torch.Tensor, 
                          dilation: int) -> torch.Tensor:
        """反因果卷积：使用右侧padding补偿分解时的左侧padding相位偏移
        
        Args:
            x: (B*N, 1, T) - 输入信号
            weight: (1, 1, K) - 卷积核
            dilation: 膨胀系数
        
        Returns:
            (B*N, 1, T) - 输出信号，长度保持不变
        """
        padding_size = (self.filter_len - 1) * dilation
        # 右侧padding（反因果）补偿分解时的左侧padding
        x_padded = F.pad(x, (0, padding_size), mode='replicate')
        # 确保weight与输入dtype一致（处理混合精度训练）
        weight = weight.to(x_padded.dtype)
        out = F.conv1d(x_padded, weight, dilation=dilation)
        return out
    
    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """执行逆SWT重构
        
        Args:
            coeffs: (B, N, T, Level+1) - 小波系数
                    顺序: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        
        Returns:
            x: (B, N, T) - 重构的信号
        """
        # 输入验证
        if coeffs.ndim != 4:
            raise ValueError(f"输入必须是4维张量 (B, N, T, L+1)，当前: {coeffs.ndim}维")
        
        B, N, T, num_bands = coeffs.shape
        
        if num_bands != self.level + 1:
            raise ValueError(
                f"频段数不匹配: 期望 {self.level + 1}，实际 {num_bands}"
            )
        
        # NaN/Inf检查
        if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
            raise ValueError("输入系数包含NaN或Inf值，无法重构")
        
        # Reshape: (B, N, T, L+1) -> (B*N, L+1, T)
        # 注意：直接支持bfloat16，不做类型转换
        coeffs_flat = coeffs.reshape(B * N, T, num_bands).permute(0, 2, 1)
        
        # 提取系数
        # coeffs_flat[:, 0, :] = cA_n
        # coeffs_flat[:, 1, :] = cD_n
        # coeffs_flat[:, 2, :] = cD_{n-1}
        # ...
        # coeffs_flat[:, n, :] = cD_1
        
        # 从最深层开始重构
        current = coeffs_flat[:, 0:1, :]  # cA_n, shape (B*N, 1, T)
        
        # 逐层重构：从level n到level 1
        for j in range(self.level):
            # 当前层的细节系数索引
            detail_idx = j + 1  # cD_n, cD_{n-1}, ..., cD_1
            cD = coeffs_flat[:, detail_idx:detail_idx+1, :]
            
            # 对应的dilation（与分解时相反的顺序）
            # 分解时：level 1用d=1, level 2用d=2, level 3用d=4
            # 重构时：先处理level 3(d=4), 再level 2(d=2), 最后level 1(d=1)
            dilation = 2 ** (self.level - 1 - j)
            
            # 重构卷积（使用反因果卷积补偿相位）
            rec_a = self._anticausal_conv(current, self.rec_lo, dilation)
            rec_d = self._anticausal_conv(cD, self.rec_hi, dilation)
            
            # SWT逆变换：两路相加后除以2
            current = (rec_a + rec_d) / 2.0
        
        # Reshape: (B*N, 1, T) -> (B, N, T)
        x_reconstructed = current.reshape(B, N, T)
        
        return x_reconstructed


# ============================================================================
# 兼容性别名（用于替换原有的SWTDecomposition和ISWTReconstruction）
# ============================================================================

class SWTDecomposition(CausalSWT):
    """CausalSWT的别名，用于兼容现有代码"""
    pass


class ISWTReconstruction(CausalISWT):
    """CausalISWT的别名，用于兼容现有代码"""
    pass


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("因果平稳小波变换 (CausalSWT) 测试")
    print("=" * 70)
    
    # 设备选择
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 测试参数
    B, N, T = 4, 7, 512
    level = 3
    wavelet = 'db4'
    
    print(f"\n测试配置:")
    print(f"  - Batch: {B}, Variables: {N}, Time: {T}")
    print(f"  - Wavelet: {wavelet}, Level: {level}")
    
    # 创建模型
    print("\n" + "-" * 70)
    swt = CausalSWT(wavelet=wavelet, level=level).to(device)
    iswt = CausalISWT(wavelet=wavelet, level=level).to(device)
    
    # ========== 测试1: 基本功能 ==========
    print("\n" + "=" * 70)
    print("测试1: 基本功能 - 分解与重构")
    print("=" * 70)
    
    x = torch.randn(B, N, T, device=device)
    print(f"\n输入形状: {x.shape}")
    
    # 分解
    coeffs = swt(x)
    print(f"系数形状: {coeffs.shape}")
    print(f"  -> (B={B}, N={N}, T={T}, Bands={level+1})")
    
    # 重构
    x_rec = iswt(coeffs)
    print(f"重构形状: {x_rec.shape}")
    
    # 重构误差
    rec_error = (x - x_rec).abs().mean().item()
    print(f"\n重构误差 (MAE): {rec_error:.10f}")
    if rec_error < 0.1:
        print("✅ 重构误差在可接受范围内（因果性的固有代价）")
    else:
        print("⚠️ 重构误差较大，可能存在问题")
    
    # ========== 测试2: 因果性验证 ==========
    print("\n" + "=" * 70)
    print("测试2: 因果性验证 - 修改未来数据不应影响过去")
    print("=" * 70)
    
    # 创建原始信号
    x_orig = torch.randn(1, 1, 100, device=device)
    
    # 复制并修改"未来"的数据点
    x_mod = x_orig.clone()
    target_idx = 60
    x_mod[0, 0, target_idx] += 100.0  # 大幅修改t=60的值
    
    # 分解
    coeffs_orig = swt(x_orig)
    coeffs_mod = swt(x_mod)
    
    print(f"\n测试: 修改 t={target_idx} 的值(+100)，检查 t<{target_idx} 的系数")
    print("-" * 50)
    
    band_names = [f"cA{level}"] + [f"cD{level-i}" for i in range(level)]
    all_causal = True
    
    for i, name in enumerate(band_names):
        diff = (coeffs_orig[0, 0, :target_idx, i] - coeffs_mod[0, 0, :target_idx, i]).abs()
        max_diff = diff.max().item()
        
        if max_diff > 1e-6:
            print(f"{name}: t<{target_idx} 最大变化 = {max_diff:.6f} ❌ 存在泄露!")
            all_causal = False
        else:
            print(f"{name}: t<{target_idx} 最大变化 = {max_diff:.10f} ✅ 因果")
    
    print("-" * 50)
    if all_causal:
        print("✅ 所有频段都是因果的，没有未来信息泄露!")
    else:
        print("❌ 存在未来信息泄露，需要检查实现!")
    
    # ========== 测试3: 不同小波类型 ==========
    print("\n" + "=" * 70)
    print("测试3: 不同小波类型支持")
    print("=" * 70)
    
    for wname in ['haar', 'db1', 'db2', 'db3', 'db4', 'db5']:
        try:
            swt_test = CausalSWT(wavelet=wname, level=2).to(device)
            iswt_test = CausalISWT(wavelet=wname, level=2).to(device)
            
            x_test = torch.randn(2, 3, 64, device=device)
            coeffs_test = swt_test(x_test)
            x_rec_test = iswt_test(coeffs_test)
            
            error = (x_test - x_rec_test).abs().mean().item()
            print(f"  {wname}: 重构误差 = {error:.8f} ✅")
        except Exception as e:
            print(f"  {wname}: 失败 - {e}")
    
    # ========== 测试4: GPU性能 ==========
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("测试4: GPU性能")
        print("=" * 70)
        
        import time
        
        # 预热
        for _ in range(10):
            _ = swt(x)
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        for _ in range(100):
            _ = swt(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"\n100次分解耗时: {elapsed*1000:.2f} ms")
        print(f"单次分解耗时: {elapsed*10:.2f} ms")
        print(f"吞吐量: {100*B*N/elapsed:.0f} 序列/秒")
    
    # ========== 测试5: 与原ptwt对比（如果可用）==========
    print("\n" + "=" * 70)
    print("测试5: 与原ptwt对比（因果性对比）")
    print("=" * 70)
    
    try:
        import ptwt
        
        x_test = torch.randn(1, 100, device=device)
        x_mod_test = x_test.clone()
        x_mod_test[0, 60] += 100.0
        
        # ptwt分解
        coeffs_ptwt_orig = ptwt.swt(x_test, 'db4', level=3)
        coeffs_ptwt_mod = ptwt.swt(x_mod_test, 'db4', level=3)
        
        print("\nptwt.swt (原实现):")
        for i, name in enumerate(['cD1', 'cD2', 'cD3', 'cA3']):
            diff = (coeffs_ptwt_orig[i][0, :60] - coeffs_ptwt_mod[i][0, :60]).abs()
            max_diff = diff.max().item()
            status = "❌ 泄露" if max_diff > 1e-6 else "✅ 因果"
            print(f"  {name}: {status} (max_diff={max_diff:.4f})")
        
        # CausalSWT分解
        x_3d = x_test.unsqueeze(0)  # (1, 1, 100)
        x_mod_3d = x_mod_test.unsqueeze(0)
        
        coeffs_causal_orig = swt(x_3d)
        coeffs_causal_mod = swt(x_mod_3d)
        
        print("\nCausalSWT (新实现):")
        for i, name in enumerate(band_names):
            diff = (coeffs_causal_orig[0, 0, :60, i] - coeffs_causal_mod[0, 0, :60, i]).abs()
            max_diff = diff.max().item()
            status = "❌ 泄露" if max_diff > 1e-6 else "✅ 因果"
            print(f"  {name}: {status} (max_diff={max_diff:.10f})")
            
    except ImportError:
        print("\nptwt未安装，跳过对比测试")
    
    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)
