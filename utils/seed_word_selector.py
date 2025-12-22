#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
种子词筛选工具

用于从词汇表中筛选出趋势相关和细节相关的种子词，用于语义筛选映射方案。
"""

import torch
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer
from utils.vocab_frequency import get_vocab_frequency_order, get_semantic_top_n_tokens


def select_trend_seed_words(tokenizer: PreTrainedTokenizer,
                            word_embeddings: torch.Tensor,
                            num_words: int = 300,
                            use_semantic_filter: bool = True) -> torch.Tensor:
    """
    筛选趋势相关的种子词
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        word_embeddings: (vocab_size, d_model) 词嵌入矩阵
        num_words: 要筛选的种子词数量
        use_semantic_filter: 是否使用语义过滤（True=语义相似度，False=仅频率）
    
    Returns:
        trend_indices: (num_words,) 趋势种子词的 token ID 索引
    """
    if use_semantic_filter:
        # 定义趋势相关的种子词汇
        trend_seed_words = [
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
            'persistent', 'sustained', 'continued', 'ongoing'
        ]
        
        trend_indices = get_semantic_top_n_tokens(
            tokenizer=tokenizer,
            n=num_words,
            word_embeddings=word_embeddings,
            method='bpe_ranks',
            seed_words=trend_seed_words
        )
    else:
        # 仅使用频率排序（取 Top-N 高频词）
        freq_indices = get_vocab_frequency_order(tokenizer, method='bpe_ranks')
        trend_indices = freq_indices[:num_words]
    
    return trend_indices


def select_detail_seed_words(tokenizer: PreTrainedTokenizer,
                             word_embeddings: torch.Tensor,
                             num_words: int = 700,
                             use_semantic_filter: bool = True) -> torch.Tensor:
    """
    筛选细节相关的种子词
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        word_embeddings: (vocab_size, d_model) 词嵌入矩阵
        num_words: 要筛选的种子词数量
        use_semantic_filter: 是否使用语义过滤（True=语义相似度，False=仅频率）
    
    Returns:
        detail_indices: (num_words,) 细节种子词的 token ID 索引
    """
    if use_semantic_filter:
        # 定义细节相关的种子词汇
        detail_seed_words = [
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
            'fluctuation', 'swing', 'oscillate', 'vibrate',
            # 异常/突发相关
            'anomaly', 'outlier', 'spike', 'drop', 'peak', 'valley',
            'surge', 'plunge', 'burst', 'explosion',
            # 局部相关
            'local', 'shortterm', 'short-term', 'temporary', 'brief',
            'momentary', 'transient', 'ephemeral',
            # 细微相关
            'subtle', 'fine', 'minute', 'tiny', 'micro', 'precise',
            'granular', 'particle', 'fragment'
        ]
        
        detail_indices = get_semantic_top_n_tokens(
            tokenizer=tokenizer,
            n=num_words,
            word_embeddings=word_embeddings,
            method='bpe_ranks',
            seed_words=detail_seed_words
        )
    else:
        # 仅使用频率排序（取 Top-N 高频词）
        freq_indices = get_vocab_frequency_order(tokenizer, method='bpe_ranks')
        detail_indices = freq_indices[:num_words]
    
    return detail_indices


def select_seed_words(tokenizer: PreTrainedTokenizer,
                      word_embeddings: torch.Tensor,
                      num_trend_words: int = 300,
                      num_detail_words: int = 700,
                      use_semantic_filter: bool = True,
                      ensure_disjoint: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    同时筛选趋势和细节种子词
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        word_embeddings: (vocab_size, d_model) 词嵌入矩阵
        num_trend_words: 趋势种子词数量
        num_detail_words: 细节种子词数量
        use_semantic_filter: 是否使用语义过滤
        ensure_disjoint: 是否确保两个词集不相交（如果启用，会调整数量以确保不重叠）
    
    Returns:
        trend_indices: (num_trend_words,) 趋势种子词的 token ID 索引
        detail_indices: (num_detail_words,) 细节种子词的 token ID 索引
    """
    trend_indices = select_trend_seed_words(
        tokenizer, word_embeddings, num_trend_words, use_semantic_filter
    )
    detail_indices = select_detail_seed_words(
        tokenizer, word_embeddings, num_detail_words, use_semantic_filter
    )
    
    if ensure_disjoint:
        # 确保两个集合不相交
        trend_set = set(trend_indices.cpu().tolist())
        detail_set = set(detail_indices.cpu().tolist())
        
        # 移除交集
        overlap = trend_set & detail_set
        if overlap:
            print(f"[seed_word_selector] ⚠️  发现 {len(overlap)} 个重叠词，正在移除...")
            
            # 从细节集合中移除重叠的词
            detail_set = detail_set - overlap
            detail_indices = torch.tensor(list(detail_set), dtype=torch.long, device=detail_indices.device)
            
            # 如果细节词数量不足，尝试补充（使用频率排序）
            if len(detail_indices) < num_detail_words:
                freq_indices = get_vocab_frequency_order(tokenizer, method='bpe_ranks')
                remaining = num_detail_words - len(detail_indices)
                # 选择不在 trend_set 和 detail_set 中的高频词
                available_indices = [idx for idx in freq_indices.cpu().tolist() 
                                   if idx not in trend_set and idx not in detail_set]
                if len(available_indices) >= remaining:
                    additional_indices = torch.tensor(available_indices[:remaining], 
                                                    dtype=torch.long, device=detail_indices.device)
                    detail_indices = torch.cat([detail_indices, additional_indices])
                    print(f"[seed_word_selector] ✅ 补充了 {remaining} 个细节词")
                else:
                    print(f"[seed_word_selector] ⚠️  无法补充足够的细节词，实际数量: {len(detail_indices)}")
        
        # 最终验证：确保两个词集不相交
        final_trend_set = set(trend_indices.cpu().tolist())
        final_detail_set = set(detail_indices.cpu().tolist())
        final_overlap = final_trend_set & final_detail_set
        
        if final_overlap:
            print(f"[seed_word_selector] ❌ 警告: 处理后仍有 {len(final_overlap)} 个重叠词")
        else:
            print(f"[seed_word_selector] ✅ 验证通过: 两个种子词集完全不相交")
            print(f"[seed_word_selector]    - 趋势种子词: {len(trend_indices)} 个（唯一）")
            print(f"[seed_word_selector]    - 细节种子词: {len(detail_indices)} 个（唯一）")
            print(f"[seed_word_selector]    - 交集大小: 0（完全隔离）")
    
    return trend_indices, detail_indices


def print_seed_words(tokenizer: PreTrainedTokenizer,
                     trend_indices: torch.Tensor,
                     detail_indices: torch.Tensor,
                     max_print: int = 50):
    """
    打印种子词信息（用于调试和验证）
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        trend_indices: 趋势种子词索引
        detail_indices: 细节种子词索引
        max_print: 每个类别最多打印的词数
    """
    print("=" * 70)
    print(f"趋势种子词 (共 {len(trend_indices)} 个，显示前 {min(max_print, len(trend_indices))} 个)")
    print("=" * 70)
    for i, token_id in enumerate(trend_indices[:max_print]):
        token_id_int = token_id.item()
        token = tokenizer.convert_ids_to_tokens([token_id_int])[0]
        decoded = tokenizer.decode([token_id_int])
        print(f"  {i+1:4d}: id={token_id_int:5d}, token='{token}', decoded='{decoded[:40]}'")
    
    print("\n" + "=" * 70)
    print(f"细节种子词 (共 {len(detail_indices)} 个，显示前 {min(max_print, len(detail_indices))} 个)")
    print("=" * 70)
    for i, token_id in enumerate(detail_indices[:max_print]):
        token_id_int = token_id.item()
        token = tokenizer.convert_ids_to_tokens([token_id_int])[0]
        decoded = tokenizer.decode([token_id_int])
        print(f"  {i+1:4d}: id={token_id_int:5d}, token='{token}', decoded='{decoded[:40]}'")
    
    # 检查重叠
    trend_set = set(trend_indices.cpu().tolist())
    detail_set = set(detail_indices.cpu().tolist())
    overlap = trend_set & detail_set
    if overlap:
        print(f"\n⚠️  警告: 发现 {len(overlap)} 个重叠词:")
        for token_id in list(overlap)[:10]:
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            decoded = tokenizer.decode([token_id])
            print(f"  id={token_id:5d}, token='{token}', decoded='{decoded[:40]}'")
    else:
        print("\n✅ 两个词集完全不相交")
    
    print("=" * 70)


if __name__ == '__main__':
    """测试脚本"""
    from transformers import GPT2Tokenizer, GPT2Model
    
    print("=" * 70)
    print("测试种子词筛选工具")
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
    
    # 测试筛选
    print("\n")
    trend_indices, detail_indices = select_seed_words(
        tokenizer=tokenizer,
        word_embeddings=word_embeddings,
        num_trend_words=300,
        num_detail_words=700,
        use_semantic_filter=True,
        ensure_disjoint=True
    )
    
    print_seed_words(tokenizer, trend_indices, detail_indices, max_print=30)
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

