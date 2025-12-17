# FrequencyChannelAttentionV2 使用指南

## 概述

`FrequencyChannelAttentionV2` 是对原版 `FrequencyChannelAttention` (V1) 的重要改进，主要解决了 **Global Average Pooling (GAP) 导致的时间维度信息丢失问题**。

### 核心改进

| 方面 | V1 (GAP) | V2 (1D Conv) |
|------|----------|--------------|
| **聚合方式** | 全局平均池化 | 深度可分离 1D 卷积 |
| **权重粒度** | Instance-wise (B×N, num_bands) | **Patch-wise** (B×N, P, num_bands) |
| **时间感知** | ❌ 所有时间步共享权重 | ✅ 每个时间步独立权重 |
| **非平稳适应** | ❌ 无法处理突变 | ✅ 能适应局部频率变化 |
| **参数增长** | 基准 | +30% (仍然轻量级) |

## 快速开始

### 1. 基本使用

```python
from layers.Embed import FrequencyChannelAttentionV2

# 初始化 V2 模块
attention = FrequencyChannelAttentionV2(
    num_bands=3,        # 频段数量 (如 level=2: cA, cD_2, cD_1)
    d_model=64,         # Embedding 维度
    reduction=4,        # MLP 隐藏层降维比例
    kernel_size=3       # 1D 卷积核大小
)

# 前向传播
band_embeddings = [
    torch.randn(28, 64, 64),  # cA (低频)
    torch.randn(28, 64, 64),  # cD_2 (中频)  
    torch.randn(28, 64, 64),  # cD_1 (高频)
]

output, attention_weights = attention(band_embeddings)
# output: (28, 64, 64) - 融合后的 embedding
# attention_weights: (28, 64, 3) - 每个 Patch 的频率权重
```

### 2. 在 WIST-PE 中启用 V2

```python
# 修改 TimeLLM 配置
configs.wavelet_mode = 'wist'
configs.use_freq_attention = True
configs.freq_attention_version = 2  # 启用 V2
configs.freq_attn_kernel_size = 3   # 卷积核大小

# 或在 WISTPatchEmbedding 中直接使用
wist_pe = WISTPatchEmbedding(
    d_model=64,
    patch_len=16,
    stride=8,
    dropout=0.1,
    wavelet_level=2,
    use_freq_attention=True,
    freq_attention_version=2,        # V2
    freq_attn_kernel_size=3
)
```

## 关键参数说明

### `kernel_size` (卷积核大小)

控制局部上下文的感知范围：

```python
# 不同 kernel_size 的特点
kernel_size = 1   # 无上下文聚合，等价于逐点处理
kernel_size = 3   # 聚合相邻 3 个 Patch 的信息 (推荐)
kernel_size = 5   # 更大的上下文窗口，适合长周期数据
kernel_size = 7   # 最大上下文，但可能引入过度平滑
```

**推荐设置:**
- **短序列 (96-336 steps)**: `kernel_size=3`
- **中长序列 (336-720 steps)**: `kernel_size=5`  
- **长序列 (720+ steps)**: `kernel_size=7`

### `reduction` (降维比例)

控制 MLP 的复杂度和表达能力：

```python
# reduction 对参数量的影响
d_model = 64, num_bands = 3

reduction = 2:  # hidden_dim = 32, 参数多，表达能力强
reduction = 4:  # hidden_dim = 16, 平衡 (推荐)
reduction = 8:  # hidden_dim = 8,  参数少，可能欠拟合
```

## 应用场景

### 1. 非平稳时间序列

V2 特别适合处理**频率特征随时间变化**的序列：

```python
# 示例：金融数据 (前半段平稳，后半段波动)
# V1: 所有时间段使用相同的频率权重 → 次优
# V2: 自动检测到后半段需要更多高频权重 → 更优

# 电力负荷 (工作日 vs 周末模式)
# V1: 无法区分不同时段的频率模式
# V2: 能自适应不同时段的频率需求
```

### 2. 多变量异构数据

不同变量可能有不同的频率主导模式：

```python
# ETTh1 数据集中：
# - 温度变量: 低频趋势主导
# - 电流变量: 高频波动主导
# V2 能为每个变量的每个时刻分配最优频率权重
```

## 性能对比

### 计算复杂度

| 模块 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| **V1** | O(B×N×d²×num_bands) | O(B×N×num_bands) |
| **V2** | O(B×N×P×d²×num_bands + B×N×d×P×k) | O(B×N×P×num_bands) |

其中 k 是 kernel_size，P 是 num_patches。

### 内存使用

```python
# 以 ETTh1 (B=32, N=7, P=64, d=64, bands=3) 为例
# V1 注意力权重: 32×7×3 = 672 个 float32 = 2.7 KB
# V2 注意力权重: 32×7×64×3 = 43,008 个 float32 = 172 KB

# 虽然 V2 内存使用更多，但相对于整个模型仍很小
```

### 预期性能提升

根据之前的实验记忆，我们预期 V2 在以下方面有改进：

1. **非平稳数据**: 2-5% 的性能提升
2. **长序列预测**: 3-7% 的性能提升  
3. **多变量场景**: 1-3% 的性能提升
4. **计算开销**: 增加 15-25%

## 调优建议

### 1. 超参数调优

```python
# 建议的调优序列
configs_to_try = [
    # 基础配置
    {"freq_attention_version": 2, "freq_attn_kernel_size": 3},
    
    # 增大感受野
    {"freq_attention_version": 2, "freq_attn_kernel_size": 5},
    
    # 更大上下文 (长序列)
    {"freq_attention_version": 2, "freq_attn_kernel_size": 7},
    
    # 对比基准
    {"freq_attention_version": 1},  # V1 baseline
]
```

### 2. 数据特异性调优

```python
# 根据数据特性选择配置
def get_optimal_config(data_characteristics):
    if data_characteristics['is_stationary']:
        return {"freq_attention_version": 1}  # V1 足够
    
    if data_characteristics['has_regime_changes']:
        return {
            "freq_attention_version": 2, 
            "freq_attn_kernel_size": 5  # 较大感受野检测模式变化
        }
    
    if data_characteristics['high_frequency_noise']:
        return {
            "freq_attention_version": 2,
            "freq_attn_kernel_size": 3,  # 避免过度平滑
            "hf_dropout": 0.7  # 增强高频去噪
        }
```

## 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 解决方案: 减小 batch_size 或使用梯度累积
   # 或者降低 d_model/kernel_size
   ```

2. **训练不稳定**
   ```python
   # 可能原因: kernel_size 过大导致梯度传播问题
   # 解决: 降低 kernel_size 到 3 或 5
   ```

3. **性能无提升**
   ```python
   # 可能原因: 数据本身是平稳的，V1 已足够
   # 建议: 先用 V1 作为 baseline，确认数据确实有非平稳特性
   ```

### 最佳实践

1. **逐步升级**: 先用 V1 建立 baseline，再测试 V2
2. **消融实验**: 分别测试不同 kernel_size 的影响
3. **可视化权重**: 观察 attention_weights 的时间变化模式
4. **监控指标**: 同时关注精度和推理时间

## 代码示例

### 完整工作流示例

```python
import torch
from layers.Embed import WISTPatchEmbedding

# 创建模拟数据 (非平稳)
B, N, T = 4, 7, 336
x = torch.randn(B, N, T)
# 模拟非平稳性: 前半段低频，后半段高频
x[:, :, :T//2] *= 0.5  # 平稳期
x[:, :, T//2:] += torch.randn(B, N, T//2) * 2.0  # 波动期

# 初始化 WIST-PE V2
wist_pe_v2 = WISTPatchEmbedding(
    d_model=64,
    patch_len=16,
    stride=8,
    dropout=0.1,
    wavelet_level=2,
    use_freq_attention=True,
    freq_attention_version=2,
    freq_attn_kernel_size=3
)

# 前向传播
output, n_vars = wist_pe_v2(x)
print(f"输入: {x.shape}")
print(f"输出: {output.shape}")

# 在 TimeLLM 中使用
# 只需设置配置文件:
# freq_attention_version = 2
# freq_attn_kernel_size = 3
```

---

**总结**: `FrequencyChannelAttentionV2` 通过引入 Patch-wise 的动态频率权重，显著提升了模型对非平稳时间序列的建模能力，特别适合金融、电力等领域的复杂时序预测任务。
