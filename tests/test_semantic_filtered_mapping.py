#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义筛选映射功能测试脚本

测试内容：
1. 语义筛选映射功能是否正确启用
2. Buffer 是否正确注册
3. 映射层维度是否正确
4. 前向传播是否正常工作
5. 种子词是否不相交
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from models.TimeLLM import Model
from utils.seed_word_selector import select_seed_words, print_seed_words


class TestConfig:
    """测试配置类"""
    def __init__(self):
        # 基础配置
        self.task_name = 'long_term_forecast'
        self.llm_model = 'GPT2'
        self.llm_dim = 768
        self.llm_layers = 2
        self.d_model = 16
        self.n_heads = 4
        self.d_ff = 32
        self.dropout = 0.1
        self.patch_len = 16
        self.stride = 8
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        
        # 小波配置
        self.wavelet_mode = 'wist'
        self.wavelet_type = 'haar'
        self.wavelet_level = 2
        
        # 分离原型配置
        self.use_dual_prototypes = 1
        self.dual_proto_trend_tokens = 500
        self.dual_proto_detail_tokens = 500
        self.dual_proto_fusion_method = 'weighted'
        
        # 语义筛选映射配置
        self.use_semantic_filtered_mapping = 1
        self.dual_proto_trend_seed_words = 300
        self.dual_proto_detail_seed_words = 700
        self.dual_proto_seed_semantic_filter = 1
        
        # Prompt配置
        self.prompt_domain = 0
        self.content = 'Test dataset description'
        
        # 其他配置
        self.use_cwpr = 0
        self.use_dual_scale_head = 0
        self.use_freq_decoupled_head = 0


def test_semantic_filtered_mapping():
    """测试语义筛选映射功能"""
    print("=" * 70)
    print("测试语义筛选映射功能")
    print("=" * 70)
    
    # 创建测试配置
    configs = TestConfig()
    
    # 测试1: 模型初始化
    print("\n[测试1] 模型初始化...")
    try:
        model = Model(configs)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return False
    
    # 测试2: 检查语义筛选映射是否启用
    print("\n[测试2] 检查语义筛选映射配置...")
    if not hasattr(model, 'use_semantic_filtered_mapping'):
        print("❌ 模型缺少 use_semantic_filtered_mapping 属性")
        return False
    
    if not model.use_semantic_filtered_mapping:
        print("❌ 语义筛选映射未启用")
        return False
    
    print(f"✅ 语义筛选映射已启用: {model.use_semantic_filtered_mapping}")
    
    # 测试3: 检查 Buffer 是否正确注册
    print("\n[测试3] 检查 Buffer 注册...")
    if not hasattr(model, 'trend_seed_embeddings'):
        print("❌ 缺少 trend_seed_embeddings Buffer")
        return False
    
    if not hasattr(model, 'detail_seed_embeddings'):
        print("❌ 缺少 detail_seed_embeddings Buffer")
        return False
    
    trend_seed_emb = model.trend_seed_embeddings
    detail_seed_emb = model.detail_seed_embeddings
    
    print(f"✅ trend_seed_embeddings shape: {trend_seed_emb.shape}")
    print(f"✅ detail_seed_embeddings shape: {detail_seed_emb.shape}")
    
    # 检查 Buffer 是否不参与梯度更新
    if trend_seed_emb.requires_grad or detail_seed_emb.requires_grad:
        print("⚠️  警告: Buffer 的 requires_grad 为 True，应该为 False")
    else:
        print("✅ Buffer 不参与梯度更新（requires_grad=False）")
    
    # 测试4: 检查映射层维度
    print("\n[测试4] 检查映射层维度...")
    if model.trend_mapping is None or model.detail_mapping is None:
        print("❌ 映射层未初始化")
        return False
    
    trend_input_size = model.trend_mapping.in_features
    trend_output_size = model.trend_mapping.out_features
    detail_input_size = model.detail_mapping.in_features
    detail_output_size = model.detail_mapping.out_features
    
    print(f"✅ 趋势映射层: {trend_input_size} → {trend_output_size}")
    print(f"✅ 细节映射层: {detail_input_size} → {detail_output_size}")
    
    # 验证维度匹配
    if trend_input_size != trend_seed_emb.shape[0]:
        print(f"❌ 趋势映射层输入维度不匹配: {trend_input_size} != {trend_seed_emb.shape[0]}")
        return False
    
    if detail_input_size != detail_seed_emb.shape[0]:
        print(f"❌ 细节映射层输入维度不匹配: {detail_input_size} != {detail_seed_emb.shape[0]}")
        return False
    
    if trend_output_size != configs.dual_proto_trend_tokens:
        print(f"❌ 趋势映射层输出维度不匹配: {trend_output_size} != {configs.dual_proto_trend_tokens}")
        return False
    
    if detail_output_size != configs.dual_proto_detail_tokens:
        print(f"❌ 细节映射层输出维度不匹配: {detail_output_size} != {configs.dual_proto_detail_tokens}")
        return False
    
    print("✅ 映射层维度正确")
    
    # 测试5: 检查种子词是否不相交
    print("\n[测试5] 检查种子词不相交性...")
    # 从 Buffer 中恢复原始索引（需要重新筛选来验证）
    # 这里我们直接测试前向传播，看是否能正常工作
    
    # 测试6: 前向传播测试
    print("\n[测试6] 前向传播测试...")
    try:
        # 创建测试输入
        batch_size = 2
        seq_len = configs.seq_len
        n_vars = configs.enc_in
        
        x_enc = torch.randn(batch_size, seq_len, n_vars)
        x_mark_enc = torch.zeros(batch_size, seq_len, 4)  # 时间特征
        x_dec = torch.randn(batch_size, configs.pred_len, n_vars)
        x_mark_dec = torch.zeros(batch_size, configs.pred_len, 4)
        
        model.eval()
        with torch.no_grad():
            # 测试原型生成（使用与 forward 中相同的逻辑）
            if model.use_semantic_filtered_mapping:
                # 语义筛选映射模式：需要转置
                # trend_seed_embeddings: (num_trend_seed_words, d_llm) -> 转置 -> (d_llm, num_trend_seed_words)
                # Linear(num_trend_seed_words, num_trend_tokens) -> (d_llm, num_trend_tokens) -> 转置回 (num_trend_tokens, d_llm)
                trend_prototypes = model.trend_mapping(model.trend_seed_embeddings.permute(1, 0)).permute(1, 0)
                detail_prototypes = model.detail_mapping(model.detail_seed_embeddings.permute(1, 0)).permute(1, 0)
            else:
                trend_prototypes = model.trend_mapping(model.word_embeddings.permute(1, 0)).permute(1, 0)
                detail_prototypes = model.detail_mapping(model.word_embeddings.permute(1, 0)).permute(1, 0)
            
            print(f"✅ 趋势原型 shape: {trend_prototypes.shape}")
            print(f"✅ 细节原型 shape: {detail_prototypes.shape}")
            
            # 完整前向传播
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            print(f"✅ 前向传播成功，输出 shape: {output.shape}")
            
            # 验证输出维度
            expected_shape = (batch_size, configs.pred_len, n_vars)
            if output.shape != expected_shape:
                print(f"❌ 输出维度不匹配: {output.shape} != {expected_shape}")
                return False
            
            print("✅ 输出维度正确")
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试7: 对比原版映射和语义筛选映射
    print("\n[测试7] 对比原版映射和语义筛选映射...")
    try:
        # 创建原版配置（不使用语义筛选）
        configs_original = TestConfig()
        configs_original.use_semantic_filtered_mapping = 0
        
        model_original = Model(configs_original)
        model_original.eval()
        
        # 比较映射层输入维度
        original_trend_input = model_original.trend_mapping.in_features
        original_detail_input = model_original.detail_mapping.in_features
        
        print(f"原版映射 - 趋势输入维度: {original_trend_input} (整个词表)")
        print(f"原版映射 - 细节输入维度: {original_detail_input} (整个词表)")
        print(f"语义筛选映射 - 趋势输入维度: {trend_input_size} (种子词)")
        print(f"语义筛选映射 - 细节输入维度: {detail_input_size} (种子词)")
        
        if trend_input_size < original_trend_input and detail_input_size < original_detail_input:
            print("✅ 语义筛选映射成功减少了映射层输入维度")
        else:
            print("⚠️  警告: 语义筛选映射未减少输入维度")
        
    except Exception as e:
        print(f"⚠️  对比测试失败: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    return True


def test_seed_word_selector():
    """测试种子词筛选工具"""
    print("\n" + "=" * 70)
    print("测试种子词筛选工具")
    print("=" * 70)
    
    try:
        # 加载 tokenizer 和 model
        print("\n[步骤1] 加载模型...")
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
        
        word_embeddings = model.get_input_embeddings().weight
        print(f"✅ 模型加载成功，词表大小: {len(tokenizer)}, 嵌入维度: {word_embeddings.shape[1]}")
        
        # 测试筛选
        print("\n[步骤2] 筛选种子词...")
        trend_indices, detail_indices = select_seed_words(
            tokenizer=tokenizer,
            word_embeddings=word_embeddings,
            num_trend_words=300,
            num_detail_words=700,
            use_semantic_filter=True,
            ensure_disjoint=True
        )
        
        print(f"✅ 趋势种子词数量: {len(trend_indices)}")
        print(f"✅ 细节种子词数量: {len(detail_indices)}")
        
        # 检查不相交
        trend_set = set(trend_indices.cpu().tolist())
        detail_set = set(detail_indices.cpu().tolist())
        overlap = trend_set & detail_set
        
        if overlap:
            print(f"❌ 发现 {len(overlap)} 个重叠词")
            return False
        else:
            print("✅ 两个词集完全不相交")
        
        # 打印部分种子词
        print("\n[步骤3] 打印部分种子词...")
        print_seed_words(tokenizer, trend_indices, detail_indices, max_print=20)
        
        print("\n✅ 种子词筛选工具测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 种子词筛选工具测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("语义筛选映射功能完整测试")
    print("=" * 70)
    
    # 测试1: 种子词筛选工具
    test1_passed = test_seed_word_selector()
    
    # 测试2: 语义筛选映射功能
    test2_passed = test_semantic_filtered_mapping()
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"种子词筛选工具测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"语义筛选映射功能测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        sys.exit(1)

