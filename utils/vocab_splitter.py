#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全词表语义切分工具

将整个词表（50k+）通过语义评分切分成趋势桶和细节桶，用于全词表映射方案。
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer


def split_full_vocab_by_semantics(
    tokenizer: PreTrainedTokenizer,
    word_embeddings: torch.Tensor,
    trend_anchors: Optional[List[str]] = None,
    detail_anchors: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将整个词表通过语义评分切分成趋势桶和细节桶
    
    基于锚点词的语义相似度，对词表中的每个词进行评分，然后通过竞价排名
    将词分配到趋势桶或细节桶。
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        word_embeddings: (vocab_size, d_llm) 词嵌入矩阵
        trend_anchors: 趋势锚点词列表（如果为None，使用默认锚点）
        detail_anchors: 细节锚点词列表（如果为None，使用默认锚点）
        verbose: 是否打印详细信息
    
    Returns:
        trend_indices: (N_trend,) 趋势词索引，LongTensor
        detail_indices: (N_detail,) 细节词索引，LongTensor
    """
    vocab_size, d_llm = word_embeddings.shape
    vocab_emb_norm = F.normalize(word_embeddings, dim=1)
    
    # 1. 定义扩展的锚点词（比种子词更多，覆盖更广）
    if trend_anchors is None:
        trend_anchors = [
            # 趋势相关
            'trend', 'pattern', 'cycle', 'period', 'seasonal',
            'upward', 'downward', 'increase', 'decrease', 'growth',
            'rising', 'falling', 'decline', 'ascend', 'descend',
            # 平滑/稳定相关
            'smooth', 'stable', 'steady', 'consistent', 'constant',
            'gradual', 'linear', 'uniform', 'even', 'regular',
            # 时间相关
            'time', 'temporal', 'longterm', 'long-term', 'chronic',
            # 统计相关（趋势类）
            'mean', 'average', 'median', 'baseline', 'level',
            # 方向相关
            'direction', 'tendency', 'drift', 'shift', 'movement',
            # 变化相关（趋势类）
            'change', 'variation', 'progression', 'evolution',
            # 持续相关
            'persistent', 'sustained', 'continued', 'ongoing',
            # 扩展：更多趋势相关词
            'trending', 'momentum', 'trajectory', 'path', 'course',
            'flow', 'stream', 'current', 'wave', 'tide'
        ]
    
    if detail_anchors is None:
        detail_anchors = [
            # 细节/波动相关
            'detail', 'fluctuation', 'oscillation', 'vibration', 'variation',
            'volatility', 'deviation', 'divergence', 'disturbance',
            # 高频/快速变化相关
            'rapid', 'fast', 'quick', 'sudden', 'abrupt', 'sharp',
            'instant', 'immediate', 'swift', 'abrupt',
            # 粗糙/不规则相关
            'rough', 'irregular', 'uneven', 'erratic', 'chaotic',
            'random', 'unstable', 'turbulent', 'noisy',
            # 小波/频域相关
            'frequency', 'spectrum', 'wavelet', 'signal', 'noise',
            'high-frequency', 'highfrequency', 'detail', 'approximation',
            # 变化相关（细节类）
            'change', 'shift', 'transition', 'movement', 'variation',
            # 扩展：更多细节相关词
            'spike', 'surge', 'jump', 'leap', 'bounce',
            'ripple', 'pulse', 'beat', 'throb', 'flutter'
        ]
    
    # 2. 收集锚点词的 token IDs
    trend_ids = []
    detail_ids = []
    
    for word in trend_anchors:
        try:
            ids = tokenizer.encode(word, add_special_tokens=False)
            trend_ids.extend(ids)
        except:
            continue
    
    for word in detail_anchors:
        try:
            ids = tokenizer.encode(word, add_special_tokens=False)
            detail_ids.extend(ids)
        except:
            continue
    
    # 去重并过滤无效ID
    trend_ids = list(set([idx for idx in trend_ids if 0 <= idx < vocab_size]))
    detail_ids = list(set([idx for idx in detail_ids if 0 <= idx < vocab_size]))
    
    if len(trend_ids) == 0 or len(detail_ids) == 0:
        raise ValueError(f"锚点词收集失败：趋势锚点 {len(trend_ids)} 个，细节锚点 {len(detail_ids)} 个")
    
    # 3. 计算锚点中心（使用平均嵌入）
    center_t = F.normalize(word_embeddings[trend_ids].mean(0, keepdim=True), dim=1)  # (1, d_llm)
    center_d = F.normalize(word_embeddings[detail_ids].mean(0, keepdim=True), dim=1)  # (1, d_llm)
    
    # 4. 全量打分（竞价排名）
    # 计算每个词到两个中心的余弦相似度
    score_t = torch.matmul(vocab_emb_norm, center_t.t()).squeeze()  # (vocab_size,)
    score_d = torch.matmul(vocab_emb_norm, center_d.t()).squeeze()  # (vocab_size,)
    
    # 5. 竞价切分：每个词归入得分更高的桶
    mask_trend = score_t > score_d
    trend_indices = torch.where(mask_trend)[0].long()
    detail_indices = torch.where(~mask_trend)[0].long()
    
    if verbose:
        print("=" * 70)
        print("[VocabSplitter] 全词表语义切分完成")
        print("=" * 70)
        print(f"  ├─ 词表大小: {vocab_size:,}")
        print(f"  ├─ 趋势锚点词: {len(trend_anchors)} 个 → {len(trend_ids)} 个有效 token")
        print(f"  ├─ 细节锚点词: {len(detail_anchors)} 个 → {len(detail_ids)} 个有效 token")
        print(f"  ├─ 趋势桶大小: {len(trend_indices):,} ({len(trend_indices)/vocab_size*100:.1f}%)")
        print(f"  ├─ 细节桶大小: {len(detail_indices):,} ({len(detail_indices)/vocab_size*100:.1f}%)")
        print(f"  └─ 切分方式: 基于余弦相似度的竞价排名")
        print("=" * 70)
    
    return trend_indices, detail_indices


def print_vocab_split_samples(
    tokenizer: PreTrainedTokenizer,
    trend_indices: torch.Tensor,
    detail_indices: torch.Tensor,
    max_print: int = 20
):
    """
    打印切分后的词表样本
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        trend_indices: 趋势词索引
        detail_indices: 细节词索引
        max_print: 每个桶最多打印的词数
    """
    print("\n" + "=" * 70)
    print("切分结果样本（前 {} 个词）".format(max_print))
    print("=" * 70)
    
    print("\n📈 趋势桶样本:")
    trend_list = trend_indices.cpu().tolist()[:max_print]
    for i, idx in enumerate(trend_list, 1):
        try:
            word = tokenizer.decode([idx])
            print(f"  {i:2d}. [{idx:5d}] {word}")
        except:
            print(f"  {i:2d}. [{idx:5d}] <decode_error>")
    
    print("\n📊 细节桶样本:")
    detail_list = detail_indices.cpu().tolist()[:max_print]
    for i, idx in enumerate(detail_list, 1):
        try:
            word = tokenizer.decode([idx])
            print(f"  {i:2d}. [{idx:5d}] {word}")
        except:
            print(f"  {i:2d}. [{idx:5d}] <decode_error>")
    
    print("=" * 70)


if __name__ == '__main__':
    """测试脚本"""
    from transformers import GPT2Tokenizer, GPT2Model
    
    print("=" * 70)
    print("测试全词表语义切分工具")
    print("=" * 70)
    
    # 加载 tokenizer 和 model
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(
            'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=True
        )
        model = GPT2Model.from_pretrained(
            'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=True
        )
        print("✅ 从本地加载模型成功")
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(
            'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=False
        )
        model = GPT2Model.from_pretrained(
            'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=False
        )
        print("✅ 下载并加载模型成功")
    
    word_embeddings = model.get_input_embeddings().weight
    print(f"✅ 词表大小: {len(tokenizer):,}, 嵌入维度: {word_embeddings.shape[1]}")
    
    # 测试切分
    print("\n[步骤1] 执行全词表语义切分...")
    trend_indices, detail_indices = split_full_vocab_by_semantics(
        tokenizer=tokenizer,
        word_embeddings=word_embeddings,
        trend_anchors=None,  # 使用默认锚点
        detail_anchors=None,
        verbose=True
    )
    
    # 打印样本
    print("\n[步骤2] 打印切分结果样本...")
    print_vocab_split_samples(tokenizer, trend_indices, detail_indices, max_print=30)
    
    # 验证不相交
    trend_set = set(trend_indices.cpu().tolist())
    detail_set = set(detail_indices.cpu().tolist())
    overlap = trend_set & detail_set
    
    if overlap:
        print(f"\n❌ 发现 {len(overlap)} 个重叠词（不应该发生）")
    else:
        print("\n✅ 两个词集完全不相交")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

