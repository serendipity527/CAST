#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
词汇频率工具函数

用于获取 tokenizer 词汇表的真实频率排序，支持基于 BPE ranks 的频率推断。
"""

import torch
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer


def get_vocab_frequency_order(tokenizer: PreTrainedTokenizer, 
                              method: str = 'bpe_ranks') -> torch.Tensor:
    """
    获取词汇表的频率排序索引
    
    对于 GPT2 等 BPE tokenizer，词汇表顺序不完全等于词频顺序。
    此函数通过分析 BPE 合并顺序（bpe_ranks）来推断真实的词频排序。
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        method: 排序方法
            - 'bpe_ranks': 基于 BPE 合并顺序推断（推荐，适用于 GPT2）
            - 'vocab_order': 使用原始词汇表顺序（不推荐，仅作fallback）
    
    Returns:
        sorted_indices: (vocab_size,) 按频率从高到低排序的 token ID 索引
    """
    vocab_size = len(tokenizer)
    
    if method == 'bpe_ranks':
        # 方法1: 基于 BPE ranks 推断频率
        if hasattr(tokenizer, 'bpe_ranks') and tokenizer.bpe_ranks:
            return _get_frequency_from_bpe_ranks(tokenizer, vocab_size)
        else:
            print(f"[vocab_frequency] ⚠️  tokenizer 没有 bpe_ranks，回退到 vocab_order 方法")
            method = 'vocab_order'
    
    if method == 'vocab_order':
        # 方法2: 使用原始词汇表顺序（不准确，但作为fallback）
        # 注意：这实际上就是简单的 0, 1, 2, ..., vocab_size-1
        return torch.arange(vocab_size)
    
    else:
        raise ValueError(f"未知的排序方法: {method}")


def _get_frequency_from_bpe_ranks(tokenizer: PreTrainedTokenizer, 
                                   vocab_size: int) -> torch.Tensor:
    """
    基于 BPE ranks 推断词汇频率排序
    
    BPE (Byte Pair Encoding) 的合并顺序反映了频率：
    - 越早合并的 token pair，频率越高
    - 单个字符 token 的频率通常高于复合 token
    
    策略：
    1. 单字节 token (0-255) 通常是最常用的，保持原顺序
    2. 对于 BPE 合并的 token，使用递归方法找到其构建路径的最小 BPE rank
    3. 特殊 token（如 <|endoftext|>）保持原位置
    
    Args:
        tokenizer: PreTrainedTokenizer 对象（必须有 bpe_ranks）
        vocab_size: 词汇表大小
    
    Returns:
        sorted_indices: (vocab_size,) 按频率从高到低排序的 token ID
    """
    bpe_ranks = tokenizer.bpe_ranks
    decoder = tokenizer.decoder  # id -> token 映射
    
    # 为每个 token 计算频率分数
    # 分数越低，频率越高（因为 BPE rank 越小表示越早合并，频率越高）
    token_scores = {}
    
    # 缓存：token -> 最小 BPE rank
    token_min_rank_cache = {}
    
    def get_token_min_rank(token: str) -> float:
        """递归获取 token 的最小 BPE rank"""
        if token in token_min_rank_cache:
            return token_min_rank_cache[token]
        
        # 单字符 token，返回一个很小的值（高频率）
        if len(token) == 1:
            rank = 0.0
            token_min_rank_cache[token] = rank
            return rank
        
        # 尝试所有可能的 BPE 分割方式
        min_rank = float('inf')
        
        # 方法1: 检查所有相邻字符对
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            if pair in bpe_ranks:
                min_rank = min(min_rank, bpe_ranks[pair])
        
        # 方法2: 尝试递归分割（如果 token 可以分解为已知的 BPE pairs）
        # 这是一个简化版本：检查 token 是否由已知的 BPE pairs 组成
        if min_rank == float('inf'):
            # 如果没找到直接的 pair，尝试将 token 分割为两部分
            for split_pos in range(1, len(token)):
                left = token[:split_pos]
                right = token[split_pos:]
                left_rank = get_token_min_rank(left)
                right_rank = get_token_min_rank(right)
                # 检查左右两部分是否可以合并
                if (left, right) in bpe_ranks:
                    combined_rank = bpe_ranks[(left, right)]
                    min_rank = min(min_rank, max(left_rank, right_rank, combined_rank))
        
        # 如果还是没找到，使用一个基于 token ID 的 fallback
        if min_rank == float('inf'):
            # 检查是否是特殊 token（通常以 <| 开头）
            if token.startswith('<|') and token.endswith('|>'):
                # 特殊 token，给一个中等优先级
                min_rank = 50000.0
            else:
                # 其他未知 token，使用一个很大的值
                min_rank = 100000.0
        
        token_min_rank_cache[token] = min_rank
        return min_rank
    
    # 为每个 token 计算分数
    for token_id in range(vocab_size):
        token = decoder.get(token_id, '')
        
        if not token:
            token_scores[token_id] = float('inf')
            continue
        
        # 获取 token 的最小 BPE rank
        min_rank = get_token_min_rank(token)
        
        # 对于单字符 token，进一步优化：常见字符（空格、字母、数字）优先级更高
        if len(token) == 1:
            char_code = ord(token[0])
            # ASCII 常见字符范围：0-127，特别是 32-126（可打印字符）
            if 32 <= char_code <= 126:
                # 可打印字符，使用更小的分数
                token_scores[token_id] = min_rank + char_code * 0.001
            else:
                # 其他单字符
                token_scores[token_id] = min_rank + 1000.0
        else:
            # 复合 token，使用最小 BPE rank
            token_scores[token_id] = min_rank
    
    # 按分数排序（分数越小，频率越高）
    sorted_pairs = sorted(token_scores.items(), key=lambda x: x[1])
    sorted_indices = torch.tensor([token_id for token_id, _ in sorted_pairs], dtype=torch.long)
    
    return sorted_indices


def get_top_n_tokens(tokenizer: PreTrainedTokenizer, 
                     n: int, 
                     method: str = 'bpe_ranks',
                     word_embeddings: Optional[torch.Tensor] = None,
                     semantic_filter: bool = False,
                     seed_words: Optional[List[str]] = None) -> torch.Tensor:
    """
    获取 Top-N 最常用 token 的索引
    
    支持基于语义相似度的过滤，选择与时间序列/小波特征相关的词汇。
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        n: 要获取的 token 数量
        method: 排序方法（见 get_vocab_frequency_order）
        word_embeddings: (vocab_size, d_model) 词嵌入矩阵，用于语义相似度计算
        semantic_filter: 是否启用语义过滤（选择与时间序列相关的词汇）
        seed_words: 种子词汇列表，用于语义相似度计算（如果为None，使用默认的时间序列相关词汇）
    
    Returns:
        top_n_indices: (n,) Top-N token 的 ID 索引
    """
    vocab_size = len(tokenizer)
    n = min(n, vocab_size)
    
    if semantic_filter and word_embeddings is not None:
        # 使用语义相似度过滤
        return get_semantic_top_n_tokens(
            tokenizer, n, word_embeddings, 
            method=method, seed_words=seed_words
        )
    else:
        # 使用频率排序
        sorted_indices = get_vocab_frequency_order(tokenizer, method=method)
        return sorted_indices[:n]


def get_semantic_top_n_tokens(tokenizer: PreTrainedTokenizer,
                              n: int,
                              word_embeddings: torch.Tensor,
                              method: str = 'bpe_ranks',
                              seed_words: Optional[List[str]] = None) -> torch.Tensor:
    """
    基于语义相似度获取 Top-N 与时间序列相关的 token
    
    策略：
    1. 定义与时间序列相关的种子词汇（trend, pattern, cycle, etc.）
    2. 计算所有词汇与种子词汇的语义相似度
    3. 结合词频和语义相似度，选择最相关的词汇
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        n: 要获取的 token 数量
        word_embeddings: (vocab_size, d_model) 词嵌入矩阵
        method: 频率排序方法
        seed_words: 种子词汇列表（如果为None，使用默认列表）
    
    Returns:
        top_n_indices: (n,) Top-N 语义相关的 token 索引
    """
    vocab_size = len(tokenizer)
    device = word_embeddings.device
    
    # 默认种子词汇：与时间序列、趋势、周期性、波动相关的词汇
    if seed_words is None:
        seed_words = [
            # 趋势相关
            'trend', 'pattern', 'cycle', 'period', 'seasonal',
            'upward', 'downward', 'increase', 'decrease', 'growth',
            # 时间相关
            'time', 'temporal', 'sequence', 'series',
            # 统计相关
            'mean', 'average', 'median', 'variance', 'volatility',
            'fluctuation', 'oscillation', 'wave', 'frequency',
            # 变化相关
            'change', 'shift', 'transition', 'movement', 'variation',
            # 小波/频域相关
            'frequency', 'spectrum', 'wavelet', 'signal', 'noise',
            'smooth', 'rough', 'detail', 'approximation',
            # 预测/动作类（新增）
            'predict', 'forecast', 'estimate', 'future', 'history', 'past', 'lag', 'horizon',
            # 程度/形容词（新增）
            'rapid', 'slow', 'stable', 'unstable', 'sharp', 'flat', 'linear', 'nonlinear',
            # 异常检测类（新增）
            'anomaly', 'outlier', 'spike', 'drop', 'normal', 'abnormal'
        ]
    
    # 获取种子词汇的嵌入
    seed_indices = []
    seed_embeddings_list = []
    
    for seed_word in seed_words:
        try:
            # 尝试直接编码
            seed_token_ids = tokenizer.encode(seed_word, add_special_tokens=False)
            if seed_token_ids:
                # 取第一个 token（通常是完整的词）
                seed_idx = seed_token_ids[0]
                if seed_idx < vocab_size:
                    seed_indices.append(seed_idx)
                    seed_embeddings_list.append(word_embeddings[seed_idx])
        except:
            continue
    
    if not seed_embeddings_list:
        print("[vocab_frequency] ⚠️  警告: 无法找到种子词汇，回退到频率排序")
        return get_top_n_tokens(tokenizer, n, method=method, semantic_filter=False)
    
    # ========== 优化1: 使用 Max-Similarity 而非 Mean-Similarity ==========
    # 计算每个候选词与所有种子词的相似度，取最大值
    # 这样可以保留多样性：只要接近任意一个种子词（如"trend"或"frequency"），就会被选中
    seed_embeddings_tensor = torch.stack(seed_embeddings_list, dim=0)  # (num_seeds, d_model)
    seed_embeddings_norm = seed_embeddings_tensor / (seed_embeddings_tensor.norm(dim=1, keepdim=True) + 1e-8)
    
    # 归一化所有词嵌入
    word_embeddings_norm = word_embeddings / (word_embeddings.norm(dim=1, keepdim=True) + 1e-8)
    
    # 计算相似度矩阵: (vocab_size, num_seeds)
    sim_matrix = torch.matmul(word_embeddings_norm, seed_embeddings_norm.T)
    
    # 对每个候选词，取其与所有种子词的最大相似度（Max-Similarity）
    max_similarities, _ = sim_matrix.max(dim=1)  # (vocab_size,)
    
    # ========== 优化2: 混合打分策略 ==========
    # 策略：先取频率 Top-20k 作为候选池，然后在候选池里按相似度排序取 Top-N
    # 这样既保证了词是 LLM 熟悉的（高频），又保证了是时序相关的（高相似度）
    freq_indices = get_vocab_frequency_order(tokenizer, method=method)
    
    # 候选池大小：取频率 Top-20k（或更大，确保覆盖足够多的高频词）
    candidate_pool_size = min(20000, vocab_size)
    candidate_pool = freq_indices[:candidate_pool_size]
    
    # 在候选池中，按最大相似度排序
    candidate_similarities = max_similarities[candidate_pool]
    
    # 选择候选池中相似度最高的 N 个
    _, top_sim_indices_in_pool = torch.topk(candidate_similarities, min(n, len(candidate_pool)), largest=True)
    top_n_indices = candidate_pool[top_sim_indices_in_pool]
    
    return top_n_indices


def print_top_n_tokens(tokenizer: PreTrainedTokenizer, 
                       n: int = 50,
                       method: str = 'bpe_ranks',
                       word_embeddings: Optional[torch.Tensor] = None,
                       semantic_filter: bool = False):
    """
    打印 Top-N 最常用 token（用于调试和验证）
    
    Args:
        tokenizer: PreTrainedTokenizer 对象
        n: 要打印的 token 数量
        method: 排序方法
        word_embeddings: 词嵌入矩阵（用于语义过滤）
        semantic_filter: 是否启用语义过滤
    """
    top_n_indices = get_top_n_tokens(
        tokenizer, n, method, 
        word_embeddings=word_embeddings,
        semantic_filter=semantic_filter
    )
    
    method_desc = f"{method}方法" + (" + 语义过滤" if semantic_filter else "")
    print("=" * 70)
    print(f"Top-{n} Token (基于 {method_desc})")
    print("=" * 70)
    
    for i, token_id in enumerate(top_n_indices):
        token_id_int = token_id.item()
        token = tokenizer.convert_ids_to_tokens([token_id_int])[0]
        decoded = tokenizer.decode([token_id_int])
        print(f"  {i+1:4d}: id={token_id_int:5d}, token='{token}', decoded='{decoded[:30]}'")
    
    print("=" * 70)


if __name__ == '__main__':
    """测试脚本"""
    from transformers import GPT2Tokenizer
    
    print("=" * 70)
    print("测试词汇频率工具")
    print("=" * 70)
    
    # 加载 tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(
            'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=True
        )
        print("✅ 从本地加载 tokenizer 成功")
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(
            'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=False
        )
        print("✅ 下载并加载 tokenizer 成功")
    
    # 测试 Top-50
    print("\n")
    print_top_n_tokens(tokenizer, n=50, method='bpe_ranks')
    
    # 对比原始词汇表顺序
    print("\n")
    print("=" * 70)
    print("对比: 原始词汇表前50个 token")
    print("=" * 70)
    for i in range(50):
        token = tokenizer.convert_ids_to_tokens([i])[0]
        decoded = tokenizer.decode([i])
        print(f"  {i+1:4d}: id={i:5d}, token='{token}', decoded='{decoded[:30]}'")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

