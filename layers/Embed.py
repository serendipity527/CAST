import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x).to(x.device)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class SoftThreshold(nn.Module):
    """
    可学习软阈值去噪模块
    公式: y = sign(x) * ReLU(|x| - tau)
    当 |x| < tau 时，输出为 0（视为噪声）
    当 |x| >= tau 时，输出为 sign(x) * (|x| - tau)（保留但收缩）
    """
    def __init__(self, num_features, init_tau=0.1):
        super(SoftThreshold, self).__init__()
        # 可学习的阈值参数，每个特征维度一个
        self.tau = nn.Parameter(torch.ones(num_features) * init_tau)
    
    def forward(self, x):
        # x: (..., num_features)
        # tau 需要 broadcast 到 x 的形状
        tau = torch.abs(self.tau)  # 确保阈值为正
        return torch.sign(x) * torch.relu(torch.abs(x) - tau)


class WaveletPatchEmbedding(nn.Module):
    """
    多分辨率 Patch Embedding：基于 Haar 小波分解
    将每个 Patch 分解为低频（趋势）和高频（细节）分量，
    分别投影后通过门控机制融合，保留显式的频域信息。
    """

    def __init__(self, d_model, patch_len, stride, dropout, use_soft_threshold=False):
        super(WaveletPatchEmbedding, self).__init__()
        # Patching 参数
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # 确保 patch_len 是偶数（Haar 小波要求）
        assert patch_len % 2 == 0, f"patch_len must be even for Haar DWT, got {patch_len}"

        # 小波分解后的长度
        self.half_len = patch_len // 2

        # 双通道独立投影：低频和高频各自有专属的 Embedding 层
        self.approx_embedding = TokenEmbedding(self.half_len, d_model)  # 低频/趋势
        self.detail_embedding = TokenEmbedding(self.half_len, d_model)  # 高频/细节

        # 门控融合机制：学习如何动态加权低频和高频特征
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # 初始化 Gate 权重：偏向低频（防止高频过拟合）
        # bias=2.0 -> Sigmoid(2.0) ≈ 0.88
        # 初始融合 = 88% 低频 (Trend) + 12% 高频 (Detail)
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 2.0)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 【优化1】高频通道专用 Dropout：防止对噪声过拟合
        # 比常规 dropout 更强 (0.5)，强迫模型学习高频的统计分布而非具体噪声
        self.detail_dropout = nn.Dropout(0.5)
        
        # 【优化2】可学习软阈值去噪：智能过滤高频噪声
        self.use_soft_threshold = use_soft_threshold
        if use_soft_threshold:
            # 对小波域的高频分量应用软阈值
            # init_tau=0.1 是初始阈值，模型会自动学习最佳值
            self.soft_threshold = SoftThreshold(num_features=self.half_len, init_tau=0.1)

        # 打印配置日志
        self._print_config()

    def _print_config(self):
        """打印当前模块的配置信息"""
        print("=" * 60)
        print("[WaveletPatchEmbedding] 小波多分辨率 Patch Embedding 已启用")
        print("=" * 60)
        print(f"  ├─ Patch 长度: {self.patch_len}")
        print(f"  ├─ Stride: {self.stride}")
        print(f"  ├─ 输出维度: {self.d_model}")
        print(f"  ├─ 小波分解: Haar DWT (单级)")
        print(f"  ├─ 低频分量长度: {self.half_len}")
        print(f"  ├─ 高频分量长度: {self.half_len}")
        print(f"  ├─ 门控初始化: 偏向低频 (Trend ~88%, Detail ~12%)")
        print(f"  ├─ 高频 Dropout: p=0.5 (防过拟合)")
        if self.use_soft_threshold:
            print(f"  ├─ 软阈值去噪: ✅ 启用 (可学习阈值)")
        else:
            print(f"  ├─ 软阈值去噪: ❌ 关闭")
        print(f"  └─ 输出 Dropout: p={self.dropout.p}")
        print("=" * 60)

    def haar_dwt_1d(self, x):
        """
        对最后一个维度执行单级 Haar 离散小波变换 (DWT)
        
        Args:
            x: 输入张量，形状 (B, num_patches, patch_len)
        
        Returns:
            approx: 近似分量（低频/趋势），形状 (B, num_patches, patch_len//2)
            detail: 细节分量（高频/波动），形状 (B, num_patches, patch_len//2)
        """
        # Haar 小波：相邻点求平均 → 低频近似
        approx = (x[..., 0::2] + x[..., 1::2]) / math.sqrt(2)
        # Haar 小波：相邻点求差 → 高频细节
        detail = (x[..., 0::2] - x[..., 1::2]) / math.sqrt(2)
        return approx, detail

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 (B, N, T)，其中 N 是变量数，T 是序列长度
        
        Returns:
            output: 融合后的 Patch Embedding，形状 (B*N, num_patches, d_model)
            n_vars: 变量数 N
        """
        # 记录变量数
        n_vars = x.shape[1]

        # Step 1: Padding 并切分为 Patches
        x = self.padding_patch_layer(x)
        # unfold: (B, N, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # reshape: (B*N, num_patches, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Step 2: Haar 小波分解
        # approx, detail: 各为 (B*N, num_patches, patch_len//2)
        approx, detail = self.haar_dwt_1d(x)
        
        # 【优化2】对高频分量应用可学习软阈值去噪
        if self.use_soft_threshold:
            detail = self.soft_threshold(detail)

        # Step 3: 双通道独立投影
        # TokenEmbedding 输入 (B, L, C)，输出 (B, L, d_model)
        # 这里 C = patch_len//2，L = num_patches
        e_approx = self.approx_embedding(approx)  # (B*N, num_patches, d_model)
        e_detail = self.detail_embedding(detail)  # (B*N, num_patches, d_model)
        
        # 【优化1】对高频分量施加强 Dropout，抑制噪声过拟合
        e_detail = self.detail_dropout(e_detail)

        # Step 4: 门控融合
        # 拼接低频和高频 embedding
        combined = torch.cat([e_approx, e_detail], dim=-1)  # (B*N, num_patches, d_model*2)
        # 计算门控权重 (0~1)，决定低频和高频的混合比例
        gate_weight = self.gate(combined)  # (B*N, num_patches, d_model)
        # 加权融合：gate * 低频 + (1-gate) * 高频
        output = gate_weight * e_approx + (1 - gate_weight) * e_detail

        return self.dropout(output), n_vars


class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class WISTPatchEmbedding(nn.Module):
    """
    WIST-PE: Wavelet-Informed Spatio-Temporal Patch Embedding
    
    核心创新点:
    1. 全局因果小波分解 (Global Causal DWT): 在 Patching 之前先做全局 db4 分解
    2. 双通道差异化处理: 低频直接投影，高频经过软阈值去噪+强Dropout
    3. 门控融合 (Gated Fusion): 偏置初始化使模型初期关注低频趋势 (88%/12%)
    4. 严格因果性: 使用 CausalSWT，仅左侧填充，防止未来信息泄露
    
    流程:
    输入 (B, N, T) 
        → 全局因果小波分解 → [低频 Trend, 高频 Detail]
        → 分别切分 Patch
        → 差异化投影 (低频直投, 高频去噪+投影+Dropout)
        → 门控融合
        → 输出 (B*N, num_patches, d_model)
    """
    
    def __init__(self, d_model, patch_len, stride, dropout,
                 wavelet_type='db4', wavelet_level=1,
                 hf_dropout=0.5, gate_bias_init=2.0,
                 use_soft_threshold=True):
        super(WISTPatchEmbedding, self).__init__()
        
        # 基础参数
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        
        # 导入因果小波变换模块
        from layers.CausalWavelet import CausalSWT
        self.swt = CausalSWT(wavelet=wavelet_type, level=wavelet_level)
        
        # Patching 层
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        
        # 双通道独立投影 (使用纯线性投影保证因果性，避免 Conv1d circular padding 导致的信息泄露)
        # 低频通道: 直接线性投影
        self.low_freq_embedding = nn.Linear(patch_len, d_model)
        # 高频通道: 线性投影
        self.high_freq_embedding = nn.Linear(patch_len, d_model)
        
        # 初始化线性层
        nn.init.kaiming_normal_(self.low_freq_embedding.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.high_freq_embedding.weight, mode='fan_in', nonlinearity='leaky_relu')
        
        # 高频通道专用 Dropout (防止对噪声过拟合)
        self.hf_dropout = nn.Dropout(hf_dropout)
        
        # 可学习软阈值去噪 (应用于高频分量)
        self.use_soft_threshold = use_soft_threshold
        if use_soft_threshold:
            self.soft_threshold = SoftThreshold(num_features=patch_len, init_tau=0.1)
        
        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 初始化门控偏置: 偏向低频
        # sigmoid(2.0) ≈ 0.88 → 初始融合 = 88% 低频 + 12% 高频
        self.gate_bias_init = gate_bias_init
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, gate_bias_init)
        
        # 输出 Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 打印配置信息
        self._print_config()
    
    def _print_config(self):
        """打印模块配置"""
        print("=" * 70)
        print("[WIST-PE] Wavelet-Informed Spatio-Temporal Patch Embedding 已启用")
        print("=" * 70)
        print(f"  ├─ 小波基类型: {self.wavelet_type}")
        print(f"  ├─ 分解层数: {self.wavelet_level}")
        print(f"  ├─ Patch 长度: {self.patch_len}")
        print(f"  ├─ Stride: {self.stride}")
        print(f"  ├─ 输出维度: {self.d_model}")
        print(f"  ├─ 高频 Dropout: p={self.hf_dropout.p}")
        print(f"  ├─ 门控初始化: bias={self.gate_bias_init:.1f} (低频≈{100*torch.sigmoid(torch.tensor(self.gate_bias_init)).item():.0f}%)")
        if self.use_soft_threshold:
            print(f"  ├─ 软阈值去噪: ✅ 启用 (可学习阈值)")
        else:
            print(f"  ├─ 软阈值去噪: ❌ 关闭")
        print(f"  └─ 特性: 全局因果小波分解 + 双通道差异化 + 门控融合")
        print("=" * 70)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 (B, N, T)，其中 N 是变量数，T 是序列长度
        
        Returns:
            output: 融合后的 Patch Embedding，形状 (B*N, num_patches, d_model)
            n_vars: 变量数 N
        """
        B, N, T = x.shape
        n_vars = N
        
        # ========== Step 1: 全局因果小波分解 ==========
        # x: (B, N, T) -> swt 输出: (B, N, T, level+1)
        # level=1 时输出 2 个频段: [cA_1 (低频), cD_1 (高频)]
        coeffs = self.swt(x)
        
        # 提取低频和高频分量
        low_freq = coeffs[:, :, :, 0]   # cA: (B, N, T) 低频/趋势
        high_freq = coeffs[:, :, :, 1]  # cD: (B, N, T) 高频/细节
        
        # ========== Step 2: 分别切分 Patch ==========
        # 对低频分量 Patching
        low_freq = self.padding_patch_layer(low_freq)
        low_patches = low_freq.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # (B, N, num_patches, patch_len) -> (B*N, num_patches, patch_len)
        low_patches = low_patches.reshape(B * N, -1, self.patch_len)
        
        # 对高频分量 Patching
        high_freq = self.padding_patch_layer(high_freq)
        high_patches = high_freq.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        high_patches = high_patches.reshape(B * N, -1, self.patch_len)
        
        # ========== Step 3: 差异化处理 ==========
        # 低频路径: 直接投影
        e_low = self.low_freq_embedding(low_patches)  # (B*N, num_patches, d_model)
        
        # 高频路径: 软阈值去噪 → 投影 → Dropout
        if self.use_soft_threshold:
            high_patches = self.soft_threshold(high_patches)
        e_high = self.high_freq_embedding(high_patches)
        e_high = self.hf_dropout(e_high)
        
        # ========== Step 4: 门控融合 ==========
        # 拼接低频和高频 embedding
        combined = torch.cat([e_low, e_high], dim=-1)  # (B*N, num_patches, d_model*2)
        
        # 计算门控权重 (0~1)
        gate_weight = self.gate(combined)  # (B*N, num_patches, d_model)
        
        # 加权融合: gate * 低频 + (1-gate) * 高频
        output = gate_weight * e_low + (1 - gate_weight) * e_high
        
        return self.dropout(output), n_vars
