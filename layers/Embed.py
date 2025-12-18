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


class FrequencyChannelAttention(nn.Module):
    """
    频率通道注意力模块 (Frequency Channel Attention)
    
    借鉴 SE-Net (Squeeze-and-Excitation) 的思想，实现 Instance-wise 的频率权重分配。
    
    核心优势:
    1. 动态路由 (Dynamic Routing): 每个样本根据自身特性动态分配频率权重
       - 样本 A 可能是低频主导 → 自动加大低频权重
       - 样本 B 可能是高频主导 → 自动加大高频权重
    2. 自动特征选择: 如果某层分解出的全是噪声，Attention 权重自然趋近于 0
    3. 替代硬编码的 gate_bias_init，实现真正的自适应
    
    流程:
    输入: [e_band_0, e_band_1, ..., e_band_n]  各频段 embedding, 形状 (B*N, P, d_model)
        → Stack: (B*N, P, d_model, num_bands)
        → Squeeze: 全局平均池化 → (B*N, num_bands, d_model)
        → Excitation: MLP → (B*N, num_bands, 1)
        → Softmax: 归一化权重 → (B*N, num_bands, 1)
        → Scale: 加权求和 → (B*N, P, d_model)
    """
    
    def __init__(self, num_bands, d_model, reduction=4):
        """
        Args:
            num_bands: 频段数量 (level + 1)
            d_model: embedding 维度
            reduction: MLP 中间层的降维比例
        """
        super(FrequencyChannelAttention, self).__init__()
        
        self.num_bands = num_bands
        self.d_model = d_model
        
        # Excitation 网络: 轻量级 MLP
        # 输入: (B*N, num_bands, d_model) -> 输出: (B*N, num_bands, 1)
        hidden_dim = max(d_model // reduction, 8)  # 确保至少有 8 个隐藏单元
        
        self.excitation = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化: 让初始权重相对均匀，但略微偏向低频
        # 最后一个 Linear 的 bias 设置为 [0.5, 0, 0, ...]
        # 这样 softmax 后低频 (第 0 个) 会有略高的初始权重
        with torch.no_grad():
            # 获取最后一个 Linear 层
            last_linear = self.excitation[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
    
    def forward(self, band_embeddings):
        """
        Args:
            band_embeddings: list of tensors, 每个形状 (B*N, num_patches, d_model)
                             顺序: [e_cA, e_cD_n, e_cD_{n-1}, ..., e_cD_1]
        
        Returns:
            output: 加权融合后的 embedding, 形状 (B*N, num_patches, d_model)
            attention_weights: 注意力权重, 形状 (B*N, num_bands), 用于可视化/调试
        """
        # Stack: list of (B*N, P, d_model) -> (B*N, P, d_model, num_bands)
        stacked = torch.stack(band_embeddings, dim=-1)
        B_N, P, D, num_bands = stacked.shape
        
        # Squeeze: 全局平均池化 (在 Patch 维度上)
        # (B*N, P, d_model, num_bands) -> (B*N, d_model, num_bands) -> (B*N, num_bands, d_model)
        squeezed = stacked.mean(dim=1).permute(0, 2, 1)  # (B*N, num_bands, d_model)
        
        # Excitation: MLP 计算每个频段的重要性分数
        # (B*N, num_bands, d_model) -> (B*N, num_bands, 1)
        scores = self.excitation(squeezed)  # (B*N, num_bands, 1)
        
        # Softmax: 归一化权重，确保总和为 1
        attention_weights = F.softmax(scores, dim=1)  # (B*N, num_bands, 1)
        
        # Scale: 加权求和
        # stacked: (B*N, P, d_model, num_bands)
        # attention_weights: (B*N, num_bands, 1) -> (B*N, 1, 1, num_bands)
        weights_expanded = attention_weights.permute(0, 2, 1).unsqueeze(1)  # (B*N, 1, 1, num_bands)
        
        # 加权求和: (B*N, P, d_model, num_bands) * (B*N, 1, 1, num_bands) -> sum -> (B*N, P, d_model)
        output = (stacked * weights_expanded).sum(dim=-1)
        
        # 返回融合结果和注意力权重 (用于调试)
        return output, attention_weights.squeeze(-1)  # (B*N, P, d_model), (B*N, num_bands)


class FrequencyChannelAttentionV2(nn.Module):
    """
    频率通道注意力模块 V2 (Frequency Channel Attention with Local Context)
    
    相比 V1 的改进:
    - 使用 1D 卷积替代 Global Average Pooling (GAP)
    - 实现 Patch-wise 的动态频率权重分配
    - 每个时间步的 Patch 可以拥有不同的频率融合权重
    - 更好地处理非平稳时间序列（如突变、趋势转折点）
    
    核心优势:
    1. 时变动态路由 (Time-Varying Dynamic Routing): 
       - 第 5 个 Patch 可能是突变点 → 自动加大高频权重
       - 第 6 个 Patch 回归平稳 → 自动加大低频权重
    2. 局部上下文感知: 1D 卷积聚合相邻 Patch 的信息，而非全局平均
    3. 保留时间维度: 输出权重形状为 (B*N, P, num_bands)，而非 (B*N, num_bands)
    
    流程:
    输入: [e_band_0, e_band_1, ..., e_band_n]  各频段 embedding, 形状 (B*N, P, d_model)
        → Stack: (B*N, P, d_model, num_bands)
        → Permute: (B*N, num_bands, d_model, P)
        → 1D Conv (在 Patch 维度上): 聚合局部上下文
        → MLP: 计算每个 Patch 的频率重要性分数
        → Softmax: 归一化权重 → (B*N, P, num_bands)
        → Scale: 逐 Patch 加权求和 → (B*N, P, d_model)
    """
    
    def __init__(self, num_bands, d_model, reduction=4, kernel_size=3):
        """
        Args:
            num_bands: 频段数量 (level + 1)
            d_model: embedding 维度
            reduction: MLP 中间层的降维比例
            kernel_size: 1D 卷积核大小，控制局部上下文范围
        """
        super(FrequencyChannelAttentionV2, self).__init__()
        
        self.num_bands = num_bands
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # 隐藏层维度
        hidden_dim = max(d_model // reduction, 8)
        
        # ========== 局部上下文聚合 (替代 GAP) ==========
        # 使用 Depthwise 1D Conv 在 Patch 维度上聚合局部信息
        # 输入: (B*N * num_bands, d_model, P)
        # 输出: (B*N * num_bands, d_model, P)  -- 保留时间维度
        self.local_context = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # same padding
                groups=d_model,  # Depthwise: 每个通道独立卷积，参数量小
                bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # ========== Excitation 网络 ==========
        # 输入: (B*N, P, num_bands, d_model)
        # 输出: (B*N, P, num_bands, 1)
        self.excitation = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化: 让初始权重相对均匀
        with torch.no_grad():
            last_linear = self.excitation[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
        
        # 打印配置
        self._print_config()
    
    def _print_config(self):
        """打印模块配置"""
        print("=" * 70)
        print("[FrequencyChannelAttentionV2] Patch-wise 频率通道注意力已启用")
        print("=" * 70)
        print(f"  ├─ 频段数量: {self.num_bands}")
        print(f"  ├─ Embedding 维度: {self.d_model}")
        print(f"  ├─ 卷积核大小: {self.kernel_size} (局部上下文范围)")
        print(f"  ├─ 聚合方式: Depthwise 1D Conv (替代 GAP)")
        print(f"  └─ 输出权重: Patch-wise (每个时间步独立权重)")
        print("=" * 70)
    
    def forward(self, band_embeddings):
        """
        Args:
            band_embeddings: list of tensors, 每个形状 (B*N, num_patches, d_model)
                             顺序: [e_cA, e_cD_n, e_cD_{n-1}, ..., e_cD_1]
        
        Returns:
            output: 加权融合后的 embedding, 形状 (B*N, num_patches, d_model)
            attention_weights: 注意力权重, 形状 (B*N, num_patches, num_bands), 用于可视化/调试
        """
        # Stack: list of (B*N, P, d_model) -> (B*N, P, d_model, num_bands)
        stacked = torch.stack(band_embeddings, dim=-1)
        B_N, P, D, num_bands = stacked.shape
        
        # ========== Step 1: 局部上下文聚合 (替代 GAP) ==========
        # 对每个频段独立应用 1D Conv
        # stacked: (B*N, P, d_model, num_bands) -> (B*N, num_bands, d_model, P)
        x = stacked.permute(0, 3, 2, 1).contiguous()
        
        # Reshape 以便批量处理所有频段
        # (B*N, num_bands, d_model, P) -> (B*N * num_bands, d_model, P)
        x = x.view(B_N * num_bands, D, P)
        
        # 1D Conv: 在 Patch 维度上聚合局部上下文
        # (B*N * num_bands, d_model, P) -> (B*N * num_bands, d_model, P)
        x = self.local_context(x)
        
        # Reshape 回来
        # (B*N * num_bands, d_model, P) -> (B*N, num_bands, d_model, P)
        x = x.view(B_N, num_bands, D, P)
        
        # Permute: (B*N, num_bands, d_model, P) -> (B*N, P, num_bands, d_model)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # ========== Step 2: Excitation (计算每个 Patch 的频率权重) ==========
        # (B*N, P, num_bands, d_model) -> (B*N, P, num_bands, 1)
        scores = self.excitation(x)
        
        # ========== Step 3: Softmax 归一化 ==========
        # 在 num_bands 维度上归一化，确保每个 Patch 的频率权重和为 1
        # (B*N, P, num_bands, 1)
        attention_weights = F.softmax(scores, dim=2)
        
        # ========== Step 4: Scale (逐 Patch 加权求和) ==========
        # stacked: (B*N, P, d_model, num_bands)
        # attention_weights: (B*N, P, num_bands, 1) -> (B*N, P, 1, num_bands)
        weights_expanded = attention_weights.permute(0, 1, 3, 2)  # (B*N, P, 1, num_bands)
        
        # 加权求和: (B*N, P, d_model, num_bands) * (B*N, P, 1, num_bands) -> sum -> (B*N, P, d_model)
        output = (stacked * weights_expanded).sum(dim=-1)
        
        # 返回融合结果和注意力权重 (用于调试)
        # attention_weights: (B*N, P, num_bands, 1) -> (B*N, P, num_bands)
        return output, attention_weights.squeeze(-1)


class FrequencyChannelAttentionV3(nn.Module):
    """
    频率通道注意力模块 V3 (Global-Local Fusion / 双流机制)
    
    核心思想: Base + Residual (基准 + 残差)
    - Global Stream: GAP -> MLP -> 全局共享权重 (稳定的先验)
    - Local Stream: 1D Conv -> MLP -> Patch-wise 动态权重 (局部微调)
    - 融合: 可学习的加权求和，让模型自动平衡全局与局部的重要性
    
    优势:
    1. 抗过拟合: Global 分支限制权重自由度，防止 Local 分支对噪声过度反应
    2. 鲁棒性: 在平稳段退化为 V1，在突变段发挥 V2 的优势
    3. 残差学习: Local 只需学习对 Global 的修正，学习难度降低
    
    流程:
    输入: [e_band_0, e_band_1, ..., e_band_n]  各频段 embedding
        → Global Stream: GAP -> MLP -> W_global (B*N, 1, num_bands)
        → Local Stream: 1D Conv -> MLP -> W_local (B*N, P, num_bands)
        → Fusion: α * W_global + (1-α) * W_local (α 可学习)
        → Softmax -> Scale -> 输出 (B*N, P, d_model)
    """
    
    def __init__(self, num_bands, d_model, reduction=4, kernel_size=3):
        """
        Args:
            num_bands: 频段数量 (level + 1)
            d_model: embedding 维度
            reduction: MLP 中间层的降维比例
            kernel_size: Local 分支的 1D 卷积核大小
        """
        super(FrequencyChannelAttentionV3, self).__init__()
        
        self.num_bands = num_bands
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # 隐藏层维度
        hidden_dim = max(d_model // reduction, 8)
        
        # ========== Global Stream (全局分支) ==========
        # GAP + MLP: 提取全局频率特征
        self.global_excitation = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # ========== Local Stream (局部分支) ==========
        # Depthwise 1D Conv: 聚合局部上下文
        self.local_context = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model,
                bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # Local MLP: 计算局部频率权重
        self.local_excitation = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # ========== Fusion (可学习的融合权重) ==========
        # alpha: 控制 Global vs Local 的平衡
        # 初始化为 0.5，让模型从均衡状态开始学习
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # 初始化: 让初始权重相对均匀
        self._init_weights()
        
        # 打印配置
        self._print_config()
    
    def _init_weights(self):
        """初始化权重"""
        with torch.no_grad():
            # Global MLP 初始化
            nn.init.zeros_(self.global_excitation[-1].weight)
            nn.init.zeros_(self.global_excitation[-1].bias)
            # Local MLP 初始化
            nn.init.zeros_(self.local_excitation[-1].weight)
            nn.init.zeros_(self.local_excitation[-1].bias)
    
    def _print_config(self):
        """打印模块配置"""
        print("=" * 70)
        print("[FrequencyChannelAttentionV3] Global-Local 双流融合机制已启用")
        print("=" * 70)
        print(f"  ├─ 频段数量: {self.num_bands}")
        print(f"  ├─ Embedding 维度: {self.d_model}")
        print(f"  ├─ Local 卷积核大小: {self.kernel_size}")
        print(f"  ├─ Global Stream: GAP + MLP (Instance-wise 基准权重)")
        print(f"  ├─ Local Stream: 1D Conv + MLP (Patch-wise 微调权重)")
        print(f"  ├─ 融合方式: α * W_global + (1-α) * W_local")
        print(f"  └─ 初始 α: {self.alpha.item():.2f} (可学习)")
        print("=" * 70)
    
    def forward(self, band_embeddings):
        """
        Args:
            band_embeddings: list of tensors, 每个形状 (B*N, num_patches, d_model)
        
        Returns:
            output: 加权融合后的 embedding, 形状 (B*N, num_patches, d_model)
            attention_weights: 注意力权重, 形状 (B*N, num_patches, num_bands)
            fusion_info: dict, 包含 alpha, global_weights, local_weights 用于调试
        """
        # Stack: list of (B*N, P, d_model) -> (B*N, P, d_model, num_bands)
        stacked = torch.stack(band_embeddings, dim=-1)
        B_N, P, D, num_bands = stacked.shape
        
        # ========== Global Stream ==========
        # GAP: (B*N, P, d_model, num_bands) -> (B*N, d_model, num_bands)
        global_feat = stacked.mean(dim=1)
        
        # Permute for MLP: (B*N, d_model, num_bands) -> (B*N, num_bands, d_model)
        global_feat = global_feat.permute(0, 2, 1)
        
        # Global MLP: (B*N, num_bands, d_model) -> (B*N, num_bands, 1)
        global_scores = self.global_excitation(global_feat)
        
        # Expand to patch dimension: (B*N, num_bands, 1) -> (B*N, P, num_bands, 1)
        global_scores = global_scores.unsqueeze(1).expand(-1, P, -1, -1)
        
        # ========== Local Stream ==========
        # Permute: (B*N, P, d_model, num_bands) -> (B*N, num_bands, d_model, P)
        x = stacked.permute(0, 3, 2, 1).contiguous()
        
        # Reshape: (B*N, num_bands, d_model, P) -> (B*N * num_bands, d_model, P)
        x = x.view(B_N * num_bands, D, P)
        
        # 1D Conv: (B*N * num_bands, d_model, P) -> (B*N * num_bands, d_model, P)
        x = self.local_context(x)
        
        # Reshape back: (B*N * num_bands, d_model, P) -> (B*N, num_bands, d_model, P)
        x = x.view(B_N, num_bands, D, P)
        
        # Permute: (B*N, num_bands, d_model, P) -> (B*N, P, num_bands, d_model)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Local MLP: (B*N, P, num_bands, d_model) -> (B*N, P, num_bands, 1)
        local_scores = self.local_excitation(x)
        
        # ========== Fusion (可学习加权) ==========
        # alpha 限制在 [0, 1] 范围内
        alpha = torch.sigmoid(self.alpha)
        
        # 加权融合: α * global + (1-α) * local
        # (B*N, P, num_bands, 1)
        fused_scores = alpha * global_scores + (1 - alpha) * local_scores
        
        # ========== Softmax 归一化 ==========
        attention_weights = F.softmax(fused_scores, dim=2)
        
        # ========== Scale (逐 Patch 加权求和) ==========
        # stacked: (B*N, P, d_model, num_bands)
        # attention_weights: (B*N, P, num_bands, 1) -> (B*N, P, 1, num_bands)
        weights_expanded = attention_weights.permute(0, 1, 3, 2)
        
        # 加权求和: (B*N, P, d_model, num_bands) * (B*N, P, 1, num_bands) -> (B*N, P, d_model)
        output = (stacked * weights_expanded).sum(dim=-1)
        
        # 构建调试信息
        fusion_info = {
            'alpha': alpha.item(),
            'global_weights': F.softmax(global_scores, dim=2).squeeze(-1),  # (B*N, P, num_bands)
            'local_weights': F.softmax(local_scores, dim=2).squeeze(-1),   # (B*N, P, num_bands)
        }
        
        return output, attention_weights.squeeze(-1), fusion_info


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


class CausalConv1d(nn.Module):
    """
    因果卷积层：只使用过去的信息，不看未来
    通过左侧填充实现因果性，恢复 Patch 间的局部连通性
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # 左侧填充量
        
        # 标准 Conv1d，不使用内置 padding
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,  # 我们手动做因果填充
            bias=False
        )
        
        # Kaiming 初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        """
        Args:
            x: (B, num_patches, patch_len) - Patch 序列
        Returns:
            out: (B, num_patches, out_channels)
        """
        # x: (B, L, C) -> (B, C, L) for Conv1d
        x = x.transpose(1, 2)
        
        # 因果填充：只在左侧填充，不填充右侧
        # 这样卷积核只能看到当前和过去的 Patch
        x = F.pad(x, (self.padding, 0))  # (left_pad, right_pad)
        
        # 确保输入类型与权重类型匹配（处理混合精度训练）
        if x.dtype != self.conv.weight.dtype:
            x = x.to(self.conv.weight.dtype)
        
        # 卷积
        x = self.conv(x)
        
        # (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        return x


class WISTPatchEmbedding(nn.Module):
    """
    WIST-PE: Wavelet-Informed Spatio-Temporal Patch Embedding
    
    核心创新点:
    1. 全局因果小波分解 (Global Causal DWT): 在 Patching 之前先做全局 db4 分解
    2. 双通道差异化处理: 低频直接投影，高频经过软阈值去噪+强Dropout
    3. 门控融合 (Gated Fusion): 偏置初始化使模型初期关注低频趋势 (88%/12%)
    4. 严格因果性: 使用 CausalSWT，仅左侧填充，防止未来信息泄露
    5. 因果卷积投影 (可选): 恢复 Patch 间的局部连通性，同时保持因果性
    6. 分层金字塔融合 (Pyramid Fusion): 支持多级小波分解，从高频到低频逐级融合
    
    流程 (level=1, 双通道模式):
    输入 (B, N, T) 
        → 全局因果小波分解 → [低频 Trend, 高频 Detail]
        → 分别切分 Patch
        → 差异化投影 (低频直投, 高频去噪+投影+Dropout)
        → 门控融合
        → 输出 (B*N, num_patches, d_model)
    
    流程 (level>=2, 金字塔融合模式):
    输入 (B, N, T)
        → 全局因果小波分解 → [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        → 分别切分 Patch 并投影
        → 分层金字塔融合: cD_1 + cD_2 → D_fused → D_fused + cA → 最终输出
        → 输出 (B*N, num_patches, d_model)
    """
    
    def __init__(self, d_model, patch_len, stride, dropout,
                 wavelet_type='db4', wavelet_level=1,
                 hf_dropout=0.5, gate_bias_init=2.0,
                 use_soft_threshold=True, use_causal_conv=True,
                 pyramid_fusion=True, mf_dropout=0.3,
                 use_freq_attention=False, freq_attention_version=1,
                 freq_attn_kernel_size=3):
        super(WISTPatchEmbedding, self).__init__()
        
        # 基础参数
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.use_causal_conv = use_causal_conv
        self.pyramid_fusion = pyramid_fusion and (wavelet_level >= 2)  # 只有 level>=2 才启用金字塔融合
        self.use_freq_attention = use_freq_attention  # 是否使用频率通道注意力替代门控融合
        self.freq_attention_version = freq_attention_version  # 1=GAP版本, 2=1D Conv版本 (Patch-wise)
        self.freq_attn_kernel_size = freq_attn_kernel_size  # V2版本的卷积核大小
        
        # 导入因果小波变换模块
        from layers.CausalWavelet import CausalSWT
        self.swt = CausalSWT(wavelet=wavelet_type, level=wavelet_level)
        
        # Patching 层
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        
        # ========== 频段投影层 ==========
        # 频段数量: level + 1 (1个低频 cA + level个高频 cD)
        self.num_bands = wavelet_level + 1
        
        if self.pyramid_fusion:
            # 金字塔融合模式: 为每个频段创建独立的投影层
            # band_embeddings[0] = cA (最低频/趋势)
            # band_embeddings[1] = cD_n (最高层细节，相对较低频)
            # band_embeddings[2] = cD_{n-1}
            # ...
            # band_embeddings[n] = cD_1 (最高频细节)
            self.band_embeddings = nn.ModuleList()
            for i in range(self.num_bands):
                if use_causal_conv:
                    self.band_embeddings.append(CausalConv1d(patch_len, d_model, kernel_size=3))
                else:
                    proj = nn.Linear(patch_len, d_model)
                    nn.init.kaiming_normal_(proj.weight, mode='fan_in', nonlinearity='leaky_relu')
                    self.band_embeddings.append(proj)
            
            # 每个高频频段的 Dropout (从 cD_n 到 cD_1，Dropout 逐渐增强)
            # cD_n (中频): mf_dropout, cD_1 (最高频): hf_dropout
            self.band_dropouts = nn.ModuleList()
            self.band_dropouts.append(nn.Identity())  # cA 不做 Dropout
            for i in range(1, self.num_bands):
                # 线性插值: 从 mf_dropout 到 hf_dropout
                if wavelet_level > 1:
                    ratio = (i - 1) / (wavelet_level - 1)  # 0 到 1
                    drop_rate = mf_dropout + ratio * (hf_dropout - mf_dropout)
                else:
                    drop_rate = hf_dropout
                self.band_dropouts.append(nn.Dropout(drop_rate))
            
            # 每个高频频段的软阈值去噪 (可选)
            self.use_soft_threshold = use_soft_threshold
            if use_soft_threshold:
                self.band_thresholds = nn.ModuleList()
                self.band_thresholds.append(nn.Identity())  # cA 不做去噪
                for i in range(1, self.num_bands):
                    # 高频频段的初始阈值更大
                    if wavelet_level > 1:
                        ratio = (i - 1) / (wavelet_level - 1)
                        init_tau = 0.05 + ratio * 0.1  # 从 0.05 到 0.15
                    else:
                        init_tau = 0.1
                    self.band_thresholds.append(SoftThreshold(num_features=patch_len, init_tau=init_tau))
            
            # ========== 融合机制选择 ==========
            if use_freq_attention:
                # 使用频率通道注意力替代硬编码的门控融合
                if freq_attention_version == 3:
                    # V3: Global-Local 双流融合机制
                    self.freq_attention = FrequencyChannelAttentionV3(
                        num_bands=self.num_bands,
                        d_model=d_model,
                        reduction=4,
                        kernel_size=freq_attn_kernel_size
                    )
                elif freq_attention_version == 2:
                    # V2: 使用 1D Conv 替代 GAP，实现 Patch-wise 动态路由
                    self.freq_attention = FrequencyChannelAttentionV2(
                        num_bands=self.num_bands,
                        d_model=d_model,
                        reduction=4,
                        kernel_size=freq_attn_kernel_size
                    )
                else:
                    # V1: 使用 GAP，实现 Instance-wise 动态路由
                    self.freq_attention = FrequencyChannelAttention(
                        num_bands=self.num_bands,
                        d_model=d_model,
                        reduction=4
                    )
                self.gate_layers = None  # 不需要门控层
            else:
                # 金字塔融合门控: 从高频到低频逐级融合
                # 需要 (num_bands - 1) 个门控层
                # gate_layers[0]: cD_1 + cD_2 的融合 (如果 level >= 2)
                # gate_layers[1]: (cD_1+cD_2) + cD_3 的融合 (如果 level >= 3)
                # ...
                # gate_layers[-1]: 所有细节 + cA 的最终融合
                self.gate_layers = nn.ModuleList()
                for i in range(self.num_bands - 1):
                    gate = nn.Sequential(
                        nn.Linear(d_model * 2, d_model),
                        nn.Sigmoid()
                    )
                    # 初始化门控偏置
                    # 最后一个门控 (融合 cA) 偏向低频
                    # 其他门控 (融合细节) 相对平衡
                    if i == self.num_bands - 2:  # 最后一个门控，融合 cA
                        bias_init = gate_bias_init
                    else:  # 细节之间的融合，相对平衡
                        bias_init = 0.5  # sigmoid(0.5) ≈ 0.62
                    for m in gate.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.constant_(m.weight, 0)
                            nn.init.constant_(m.bias, bias_init)
                    self.gate_layers.append(gate)
                self.freq_attention = None  # 不使用注意力
        else:
            # 原始双通道模式 (level=1 或禁用金字塔融合)
            if use_causal_conv:
                self.low_freq_embedding = CausalConv1d(patch_len, d_model, kernel_size=3)
                self.high_freq_embedding = CausalConv1d(patch_len, d_model, kernel_size=3)
            else:
                self.low_freq_embedding = nn.Linear(patch_len, d_model)
                self.high_freq_embedding = nn.Linear(patch_len, d_model)
                nn.init.kaiming_normal_(self.low_freq_embedding.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.kaiming_normal_(self.high_freq_embedding.weight, mode='fan_in', nonlinearity='leaky_relu')
            
            self.hf_dropout = nn.Dropout(hf_dropout)
            
            self.use_soft_threshold = use_soft_threshold
            if use_soft_threshold:
                self.soft_threshold = SoftThreshold(num_features=patch_len, init_tau=0.1)
            
            # ========== 融合机制选择 ==========
            if use_freq_attention:
                # 使用频率通道注意力替代门控融合
                if freq_attention_version == 3:
                    # V3: Global-Local 双流融合机制
                    self.freq_attention = FrequencyChannelAttentionV3(
                        num_bands=2,  # 双通道: 低频 + 高频
                        d_model=d_model,
                        reduction=4,
                        kernel_size=freq_attn_kernel_size
                    )
                elif freq_attention_version == 2:
                    # V2: 使用 1D Conv 替代 GAP，实现 Patch-wise 动态路由
                    self.freq_attention = FrequencyChannelAttentionV2(
                        num_bands=2,  # 双通道: 低频 + 高频
                        d_model=d_model,
                        reduction=4,
                        kernel_size=freq_attn_kernel_size
                    )
                else:
                    # V1: 使用 GAP，实现 Instance-wise 动态路由
                    self.freq_attention = FrequencyChannelAttention(
                        num_bands=2,  # 双通道: 低频 + 高频
                        d_model=d_model,
                        reduction=4
                    )
                self.gate = None
            else:
                self.gate = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.Sigmoid()
                )
                for m in self.gate.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.constant_(m.weight, 0)
                        nn.init.constant_(m.bias, gate_bias_init)
                self.freq_attention = None
        
        # 保存参数用于打印
        self.gate_bias_init = gate_bias_init
        self.hf_dropout_rate = hf_dropout
        self.mf_dropout_rate = mf_dropout
        
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
        print(f"  ├─ 频段数量: {self.num_bands} (1个低频 + {self.wavelet_level}个高频)")
        print(f"  ├─ Patch 长度: {self.patch_len}")
        print(f"  ├─ Stride: {self.stride}")
        print(f"  ├─ 输出维度: {self.d_model}")
        if self.use_causal_conv:
            print(f"  ├─ 投影方式: ✅ 因果卷积 (CausalConv1d, kernel=3)")
        else:
            print(f"  ├─ 投影方式: Linear (无 Patch 间交互)")
        
        if self.pyramid_fusion:
            print(f"  ├─ 融合模式: ✅ 分层金字塔融合 (Pyramid Fusion)")
            print(f"  │   ├─ 中频 Dropout: p={self.mf_dropout_rate}")
            print(f"  │   ├─ 高频 Dropout: p={self.hf_dropout_rate}")
            print(f"  │   └─ 融合顺序: cD_1 → cD_2 → ... → cD_n → cA")
        else:
            print(f"  ├─ 融合模式: 双通道融合 (Dual-Channel)")
            print(f"  ├─ 高频 Dropout: p={self.hf_dropout_rate}")
        
        # 频率通道注意力 vs 门控融合
        if self.use_freq_attention:
            if self.freq_attention_version == 3:
                print(f"  ├─ 融合机制: ✅ 频率通道注意力 V3 (Global-Local 双流融合)")
                print(f"  │   ├─ Global Stream: GAP + MLP (基准权重)")
                print(f"  │   ├─ Local Stream: 1D Conv + MLP (微调权重)")
                print(f"  │   └─ 卷积核大小: {self.freq_attn_kernel_size}")
            elif self.freq_attention_version == 2:
                print(f"  ├─ 融合机制: ✅ 频率通道注意力 V2 (1D Conv, Patch-wise 动态路由)")
                print(f"  │   └─ 卷积核大小: {self.freq_attn_kernel_size}")
            else:
                print(f"  ├─ 融合机制: ✅ 频率通道注意力 V1 (GAP, Instance-wise 动态路由)")
        else:
            print(f"  ├─ 融合机制: 门控融合 (Gate Fusion)")
            print(f"  ├─ 门控初始化: bias={self.gate_bias_init:.1f} (低频≈{100*torch.sigmoid(torch.tensor(self.gate_bias_init)).item():.0f}%)")
        
        if self.use_soft_threshold:
            print(f"  ├─ 软阈值去噪: ✅ 启用 (可学习阈值)")
        else:
            print(f"  ├─ 软阈值去噪: ❌ 关闭")
        
        fusion_type = '注意力' if self.use_freq_attention else ('金字塔' if self.pyramid_fusion else '门控')
        print(f"  └─ 特性: 全局因果小波分解 + 差异化处理 + {fusion_type}融合")
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
        # 输出顺序: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        coeffs = self.swt(x)
        
        if self.pyramid_fusion:
            # ========== 金字塔融合模式 ==========
            return self._forward_pyramid(coeffs, B, N, n_vars)
        else:
            # ========== 原始双通道模式 ==========
            return self._forward_dual_channel(coeffs, B, N, n_vars)
    
    def _forward_dual_channel(self, coeffs, B, N, n_vars):
        """原始双通道融合模式 (level=1)"""
        # 提取低频和高频分量
        low_freq = coeffs[:, :, :, 0]   # cA: (B, N, T) 低频/趋势
        high_freq = coeffs[:, :, :, 1]  # cD: (B, N, T) 高频/细节
        
        # 对低频分量 Patching
        low_freq = self.padding_patch_layer(low_freq)
        low_patches = low_freq.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        low_patches = low_patches.reshape(B * N, -1, self.patch_len)
        
        # 对高频分量 Patching
        high_freq = self.padding_patch_layer(high_freq)
        high_patches = high_freq.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        high_patches = high_patches.reshape(B * N, -1, self.patch_len)
        
        # 低频路径: 直接投影
        # 确保输入类型与权重类型匹配（处理混合精度训练）
        if hasattr(self.low_freq_embedding, 'weight') and low_patches.dtype != self.low_freq_embedding.weight.dtype:
            low_patches = low_patches.to(self.low_freq_embedding.weight.dtype)
        elif hasattr(self.low_freq_embedding, 'conv') and hasattr(self.low_freq_embedding.conv, 'weight') and low_patches.dtype != self.low_freq_embedding.conv.weight.dtype:
            low_patches = low_patches.to(self.low_freq_embedding.conv.weight.dtype)
        e_low = self.low_freq_embedding(low_patches)
        
        # 高频路径: 软阈值去噪 → 投影 → Dropout
        if self.use_soft_threshold:
            high_patches = self.soft_threshold(high_patches)
        # 确保输入类型与权重类型匹配（处理混合精度训练）
        if hasattr(self.high_freq_embedding, 'weight') and high_patches.dtype != self.high_freq_embedding.weight.dtype:
            high_patches = high_patches.to(self.high_freq_embedding.weight.dtype)
        elif hasattr(self.high_freq_embedding, 'conv') and hasattr(self.high_freq_embedding.conv, 'weight') and high_patches.dtype != self.high_freq_embedding.conv.weight.dtype:
            high_patches = high_patches.to(self.high_freq_embedding.conv.weight.dtype)
        e_high = self.high_freq_embedding(high_patches)
        e_high = self.hf_dropout(e_high)
        
        # ========== 融合机制 ==========
        if self.use_freq_attention:
            # 使用频率通道注意力 (Instance-wise 动态路由)
            output, _ = self.freq_attention([e_low, e_high])
        else:
            # 门控融合
            combined = torch.cat([e_low, e_high], dim=-1)
            gate_weight = self.gate(combined)
            output = gate_weight * e_low + (1 - gate_weight) * e_high
        
        return self.dropout(output), n_vars
    
    def _forward_pyramid(self, coeffs, B, N, n_vars):
        """
        分层金字塔融合模式 (level >= 2)
        
        融合顺序 (以 level=2 为例):
        coeffs 顺序: [cA_2, cD_2, cD_1]
        
        Step 1: 对每个频段进行 Patching 和投影
        Step 2: 从最高频开始逐级融合
            - e_D1 (最高频) + e_D2 (中频) → e_detail_fused
            - e_detail_fused + e_A (低频) → e_final
        
        融合顺序 (以 level=3 为例):
        coeffs 顺序: [cA_3, cD_3, cD_2, cD_1]
        
        Step 2:
            - e_D1 + e_D2 → e_fused_12
            - e_fused_12 + e_D3 → e_detail_fused
            - e_detail_fused + e_A → e_final
        """
        # ========== Step 1: 对每个频段进行 Patching 和投影 ==========
        band_embeddings = []
        
        for i in range(self.num_bands):
            # 提取第 i 个频段
            band = coeffs[:, :, :, i]  # (B, N, T)
            
            # Patching
            band = self.padding_patch_layer(band)
            patches = band.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            patches = patches.reshape(B * N, -1, self.patch_len)
            
            # 对高频频段应用软阈值去噪 (i > 0 表示高频)
            if i > 0 and self.use_soft_threshold:
                patches = self.band_thresholds[i](patches)
            
            # 确保输入类型与权重类型匹配（处理混合精度训练）
            embedding_layer = self.band_embeddings[i]
            if hasattr(embedding_layer, 'weight') and patches.dtype != embedding_layer.weight.dtype:
                patches = patches.to(embedding_layer.weight.dtype)
            elif hasattr(embedding_layer, 'conv') and hasattr(embedding_layer.conv, 'weight') and patches.dtype != embedding_layer.conv.weight.dtype:
                patches = patches.to(embedding_layer.conv.weight.dtype)
            
            # 投影
            e_band = embedding_layer(patches)  # (B*N, num_patches, d_model)
            
            # 对高频频段应用 Dropout
            e_band = self.band_dropouts[i](e_band)
            
            band_embeddings.append(e_band)
        
        # band_embeddings 顺序: [e_cA, e_cD_n, e_cD_{n-1}, ..., e_cD_1]
        
        # ========== Step 2: 融合机制 ==========
        if self.use_freq_attention:
            # 使用频率通道注意力 (Instance-wise 动态路由)
            # 直接将所有频段传入注意力模块，让它自动学习权重
            e_fused, _ = self.freq_attention(band_embeddings)
        else:
            # 原始门控融合: 从最高频 (cD_1) 开始，逐级向低频融合
            # 最高频在 band_embeddings 的最后一个位置
            
            # 初始化: 从最高频开始
            e_fused = band_embeddings[-1]  # e_cD_1 (最高频)
            
            # 逐级融合: cD_1 → cD_2 → ... → cD_n → cA
            # 融合顺序: band_embeddings[-2], band_embeddings[-3], ..., band_embeddings[0]
            for i in range(self.num_bands - 2, -1, -1):
                e_next = band_embeddings[i]  # 下一个要融合的频段 (更低频)
                
                # 门控索引: 从 0 开始
                gate_idx = (self.num_bands - 2) - i
                
                # 门控融合
                combined = torch.cat([e_fused, e_next], dim=-1)
                gate_weight = self.gate_layers[gate_idx](combined)
                
                # 融合: gate * 当前融合结果 + (1-gate) * 下一个频段
                # 注意: 对于最后一个门控 (融合 cA)，gate 偏向低频，所以 (1-gate) 会更大
                e_fused = gate_weight * e_fused + (1 - gate_weight) * e_next
        
        return self.dropout(e_fused), n_vars
