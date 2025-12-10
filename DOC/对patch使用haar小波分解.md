



基于代码分析，以下是相比原版 TimeLLM **新增的功能模块**：

---

## 📋 新增功能梳理

### 1. 小波多分辨率 Patch Embedding (`--use_wavelet 1`)

**原版**：直接将 Patch 投影到 `d_model` 维度
```
Patch → TokenEmbedding → d_model
```

**新版**：Haar 小波分解 + 双通道投影 + 门控融合
```
Patch → Haar DWT → [低频 approx, 高频 detail]
                        ↓              ↓
                  approx_embedding  detail_embedding
                        ↓              ↓
                        └──── Gate 融合 ────┘
                                ↓
                            d_model
```

**核心组件**：
| 组件 | 功能 |
|------|------|
| [haar_dwt_1d()](cci:1://file:///home/dmx_MT/LZF/project/CAST/layers/Embed.py:282:4-297:29) | Haar 小波分解，分离趋势与细节 |
| `approx_embedding` | 低频分量投影层 |
| `detail_embedding` | 高频分量投影层 |
| `gate` | 门控融合 (Linear + Sigmoid)，动态加权 |

---

### 2. 门控偏置初始化 (88% / 12%)

**目的**：防止高频噪声过拟合

```python
# bias=2.0 → Sigmoid(2.0) ≈ 0.88
# 初始融合 = 88% 低频 (Trend) + 12% 高频 (Detail)
nn.init.constant_(m.bias, 2.0)
```

---

### 3. 高频通道 Dropout (`p=0.5`)

**目的**：强迫模型学习高频的统计规律，而非过拟合具体噪声

```python
self.detail_dropout = nn.Dropout(0.5)  # 比常规 dropout 更强
```

---

### 4. 可学习软阈值去噪 (`--use_soft_threshold 1`)

**目的**：智能过滤高频噪声，保留有意义的细节

```python
# 公式: y = sign(x) * ReLU(|x| - tau)
# tau 是可学习参数，自动学习最佳阈值
class SoftThreshold(nn.Module):
    def __init__(self, num_features, init_tau=0.1):
        self.tau = nn.Parameter(torch.ones(num_features) * init_tau)
```

---

## 🎛️ 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_wavelet` | 0 | 0=原版, 1=小波版 |
| `--use_soft_threshold` | 0 | 0=关闭, 1=开启软阈值去噪 |

---

## 📊 当前最优配置

根据您的实验记录，**最佳组合**是：
```
Haar 小波 + 双通道投影 + 88%/12% 偏置 + Dropout(0.5) + Soft Threshold
```

对应命令参数：`--use_wavelet 1 --use_soft_threshold 1`