好的，我先阅读这些文件来了解现有代码结构，然后给出实现思路。



好的，我已经阅读了所有相关文件。以下是实现 WIST-PE 方案的详细思路：

---

# WIST-PE 实现思路

## 一、现有代码分析

### 已有组件
| 文件 | 相关组件 | 状态 |
|------|----------|------|
| [CausalWavelet.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/layers/CausalWavelet.py:0:0-0:0) | [CausalSWT](cci:2://file:///home/dmx_MT/LZF/project/CAST/layers/CausalWavelet.py:122:0-261:21) (因果 db4 小波分解) | ✅ 可复用 |
| [Embed.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/layers/Embed.py:0:0-0:0) | [SoftThreshold](cci:2://file:///home/dmx_MT/LZF/project/CAST/layers/Embed.py:187:0-203:61), [WaveletPatchEmbedding](cci:2://file:///home/dmx_MT/LZF/project/CAST/layers/Embed.py:206:0-343:43) | ⚠️ 需改造 |
| [TimeLLM.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/models/TimeLLM.py:0:0-0:0) | `use_haar_wavelet` 参数控制 | ⚠️ 需扩展 |
| [run_main.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/run_main.py:0:0-0:0) | `--use_haar_wavelet` 命令行参数 | ⚠️ 需扩展 |

### 当前方案 vs 新方案对比
| 特性 | 当前 [WaveletPatchEmbedding](cci:2://file:///home/dmx_MT/LZF/project/CAST/layers/Embed.py:206:0-343:43) | 新 WIST-PE 方案 |
|------|------------------------------|-----------------|
| 分解时机 | Patch 内部（局部） | **全局分解后再 Patching** |
| 小波基 | Haar | **db4（更平滑）** |
| 因果性 | 无 | **因果卷积（仅左填充）** |
| 软阈值 | 可选 | **默认启用** |

---

## 二、新增命令行参数设计

在 [run_main.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/run_main.py:0:0-0:0) 中添加以下参数：

```python
# WIST-PE 配置参数
parser.add_argument('--wavelet_mode', type=str, default='none', 
                    choices=['none', 'haar', 'wist'],
                    help='小波嵌入模式: none=原版, haar=现有方案, wist=新WIST-PE方案')
parser.add_argument('--wavelet_type', type=str, default='db4',
                    help='小波基类型 (仅wist模式): db1/db2/db3/db4/db5/haar')
parser.add_argument('--wavelet_level', type=int, default=1,
                    help='小波分解层数 (仅wist模式): 1表示单级分解')
parser.add_argument('--hf_dropout', type=float, default=0.5,
                    help='高频通道Dropout率 (仅wist模式)')
parser.add_argument('--gate_bias_init', type=float, default=2.0,
                    help='门控融合初始偏置 (sigmoid(2.0)≈88%关注低频)')
parser.add_argument('--use_soft_threshold', type=int, default=1,
                    help='是否启用可学习软阈值去噪: 0=关闭, 1=开启')
```

**兼容性**：原有 `--use_haar_wavelet=1` 等价于 `--wavelet_mode=haar`。

---

## 三、新模块设计

### 1. 在 [Embed.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/layers/Embed.py:0:0-0:0) 中新增 `WISTPatchEmbedding` 类

```
核心流程:
输入 x: (B, N, T) 
    ↓
[1] 因果填充 (左侧 replicate padding)
    ↓
[2] 全局 CausalSWT 分解 (db4, level=1)
    ├─→ low_freq:  (B, N, T)  趋势分量
    └─→ high_freq: (B, N, T)  细节分量
    ↓
[3] 分别切分 Patch
    ├─→ low_patches:  (B*N, num_patches, patch_len)
    └─→ high_patches: (B*N, num_patches, patch_len)
    ↓
[4] 差异化处理
    ├─ 低频路: LinearProjection → e_low
    └─ 高频路: SoftThreshold → LinearProjection → Dropout(0.5) → e_high
    ↓
[5] 门控融合 (Gated Fusion)
    combined = concat(e_low, e_high)
    gate = Sigmoid(Linear(combined))  # bias_init=2.0
    output = gate * e_low + (1-gate) * e_high
    ↓
输出: (B*N, num_patches, d_model)
```

### 2. 关键实现细节

#### A. 全局因果小波分解
```python
# 复用 CausalSWT，但只取 level=1 的 cA 和 cD
from layers.CausalWavelet import CausalSWT

class WISTPatchEmbedding(nn.Module):
    def __init__(self, ...):
        self.swt = CausalSWT(wavelet=wavelet_type, level=1)
        # swt 输出: (B, N, T, 2) -> [cA_1, cD_1]
```

#### B. 软阈值去噪（复用已有 [SoftThreshold](cci:2://file:///home/dmx_MT/LZF/project/CAST/layers/Embed.py:187:0-203:61)）
```python
# 应用于高频分量
self.soft_threshold = SoftThreshold(num_features=patch_len, init_tau=0.1)
```

#### C. 门控初始化偏向低频
```python
# 初始化 gate 的 bias = 2.0
# sigmoid(2.0) ≈ 0.88 → 初始融合 = 88% 低频 + 12% 高频
for m in self.gate.modules():
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, gate_bias_init)
```

---

## 四、模型集成（TimeLLM.py）

### 修改 [Model.__init__](cci:1://file:///home/dmx_MT/LZF/project/CAST/models/TimeLLM.py:31:4-203:71)

```python
self.wavelet_mode = getattr(configs, 'wavelet_mode', 'none')

if self.wavelet_mode == 'wist':
    # 新 WIST-PE 方案
    self.patch_embedding = WISTPatchEmbedding(
        d_model=configs.d_model,
        patch_len=self.patch_len,
        stride=self.stride,
        dropout=configs.dropout,
        wavelet_type=configs.wavelet_type,
        wavelet_level=configs.wavelet_level,
        hf_dropout=configs.hf_dropout,
        gate_bias_init=configs.gate_bias_init,
        use_soft_threshold=configs.use_soft_threshold,
    )
elif self.wavelet_mode == 'haar':
    # 现有 Haar 方案（保持兼容）
    self.patch_embedding = WaveletPatchEmbedding(...)
else:
    # 原版 PatchEmbedding
    self.patch_embedding = PatchEmbedding(...)
```

---

## 五、文件修改清单

| 文件 | 修改内容 |
|------|----------|
| [run_main.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/run_main.py:0:0-0:0) | 添加 `wavelet_mode`, `wavelet_type`, `wavelet_level`, `hf_dropout`, `gate_bias_init`, `use_soft_threshold` 参数 |
| [layers/Embed.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/layers/Embed.py:0:0-0:0) | 新增 `WISTPatchEmbedding` 类 |
| [models/TimeLLM.py](cci:7://file:///home/dmx_MT/LZF/project/CAST/models/TimeLLM.py:0:0-0:0) | 根据 `wavelet_mode` 选择 Embedding 类型 |

---

## 六、命令行使用示例

```bash
# 原版方案 (无小波)
--wavelet_mode=none

# 现有 Haar 方案
--wavelet_mode=haar

# 新 WIST-PE 方案 (完整配置)
--wavelet_mode=wist \
--wavelet_type=db4 \
--wavelet_level=1 \
--hf_dropout=0.5 \
--gate_bias_init=2.0 \
--use_soft_threshold=1
```

---

## 七、设计亮点总结

1. **向后兼容**：通过 `wavelet_mode` 参数实现三种模式共存
2. **复用 [CausalSWT](cci:2://file:///home/dmx_MT/LZF/project/CAST/layers/CausalWavelet.py:122:0-261:21)**：保证因果性，避免重复造轮子
3. **模块化**：新增类不影响现有代码结构
4. **超参可调**：所有关键设计（小波基、dropout率、门控偏置）均可通过命令行控制
5. **全局→局部**：先全局分解再 Patching，更符合信号处理直觉

---

准备好后，请告诉我，我将开始实现代码。