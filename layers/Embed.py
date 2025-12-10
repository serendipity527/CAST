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
