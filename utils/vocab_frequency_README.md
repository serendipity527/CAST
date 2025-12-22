# 词汇频率工具优化说明

## 核心优化

### 1. Max-Similarity vs Mean-Similarity

**问题**：时间序列的语义是多样的。"趋势（Trend）"和"频率（Frequency）"在语义空间中可能相距较远。如果使用平均向量，得到的中心点可能既不像"趋势"也不像"频率"，而是一个平庸的中间点。

**解决方案**：使用 **Max-Similarity** 方法
- 计算每个候选词与**所有**种子词的相似度矩阵
- 对每个候选词，取其与**任意**种子词的**最大相似度**作为得分
- 只要它接近"趋势"**或者**接近"频率"，它就是我们要的词

**优势**：
- 保留语义多样性：`volatility` 可能离 `average` 很远，但用 Max 逻辑它能被选中
- 避免平均化带来的信息损失
- 更适合多语义概念的时间序列领域

### 2. 混合打分策略（Hybrid Scoring）

**策略**：先取频率 Top-20k 作为候选池，然后在候选池里按相似度排序取 Top-N

**优势**：
- 既保证了词是 LLM 熟悉的（高频词，训练充分）
- 又保证了是时序相关的（高相似度）
- 避免了选择极度生僻的专业术语（LLM 可能没训练好）

**实现**：
```python
# 1. 获取频率 Top-20k 作为候选池
candidate_pool = freq_indices[:20000]

# 2. 在候选池中，按最大相似度排序
candidate_similarities = max_similarities[candidate_pool]

# 3. 选择候选池中相似度最高的 N 个
top_n_indices = candidate_pool[top_sim_indices_in_pool]
```

### 3. 种子词库扩充

**新增类别**：
- **预测/动作类**：`predict`, `forecast`, `estimate`, `future`, `history`, `past`, `lag`, `horizon`
- **程度/形容词**：`rapid`, `slow`, `stable`, `unstable`, `sharp`, `flat`, `linear`, `nonlinear`
- **异常检测类**：`anomaly`, `outlier`, `spike`, `drop`, `normal`, `abnormal`

**覆盖范围**：
- 趋势相关：trend, pattern, cycle, period, seasonal
- 时间相关：time, temporal, sequence, series
- 统计相关：mean, average, median, variance, volatility
- 小波/频域相关：frequency, spectrum, wavelet, signal, noise
- 预测相关：predict, forecast, estimate, future, history
- 程度相关：rapid, slow, stable, sharp, flat
- 异常相关：anomaly, outlier, spike, drop

## 使用示例

```python
from utils.vocab_frequency import get_top_n_tokens
from transformers import GPT2Tokenizer, GPT2Model

# 加载模型
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
model = GPT2Model.from_pretrained('openai-community/gpt2')
word_embeddings = model.get_input_embeddings().weight

# 获取语义过滤的 Top-5000 词汇
top5000_semantic = get_top_n_tokens(
    tokenizer, 5000,
    method='bpe_ranks',
    word_embeddings=word_embeddings,
    semantic_filter=True  # 启用语义过滤
)

# 使用这些词汇进行 K-Means 初始化
candidates = word_embeddings[top5000_semantic]
# ... K-Means 聚类 ...
```

## 验证测试

运行测试脚本验证效果：

```bash
# 完整测试
python test_semantic_filter.py

# 快速测试
python quick_test_semantic.py
```

测试会验证：
1. 语义相似度提升（Max-Similarity 均值）
2. 时间序列关键词覆盖率
3. 词汇质量（Top 词汇示例）

