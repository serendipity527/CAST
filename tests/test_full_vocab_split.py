#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全词表切分功能测试脚本

测试全词表语义切分功能，包括：
1. vocab_splitter 函数测试
2. TimeLLM 模型初始化测试
3. 前向传播测试
4. 参数验证
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from utils.vocab_splitter import split_full_vocab_by_semantics, print_vocab_split_samples
from models.TimeLLM import Model


class TestConfig:
    """测试配置类"""
    def __init__(self):
        # 基础配置
        self.task_name = 'long_term_forecast'
        self.enc_in = 7  # ETTh1
        self.dec_in = 7
        self.c_out = 7
        self.seq_len = 96
        self.pred_len = 96
        self.d_model = 16
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 32
        self.dropout = 0.1
        self.activation = 'gelu'
        self.output_attention = False
        self.llm_model = 'GPT2'
        self.llm_dim = 768  # GPT2 的嵌入维度
        self.llm_layers = 2
        self.patch_len = 16
        self.stride = 8
        
        # 小波配置
        self.wavelet_mode = 'none'
        self.use_haar_wavelet = 0
        
        # Prompt配置
        self.prompt_domain = 0
        self.content = 'Test dataset description'
        
        # 分离原型配置
        self.use_dual_prototypes = 1
        self.dual_proto_trend_tokens = 1000
        self.dual_proto_detail_tokens = 1000
        self.dual_proto_fusion_method = 'mean'
        
        # 全词表切分配置
        self.use_full_vocab_split = 1
        self.use_semantic_filtered_mapping = 0  # 必须为0，因为互斥
        
        # 其他配置
        self.use_cwpr = 0
        self.use_dual_scale_head = 0
        self.use_freq_decoupled_head = 0


def test_vocab_splitter():
    """测试 vocab_splitter 函数"""
    print("\n" + "=" * 70)
    print("测试 1: vocab_splitter 函数")
    print("=" * 70)
    
    try:
        # 加载模型
        print("\n[步骤1] 加载 GPT2 模型...")
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
        print(f"✅ 模型加载成功，词表大小: {len(tokenizer):,}, 嵌入维度: {word_embeddings.shape[1]}")
        
        # 执行切分
        print("\n[步骤2] 执行全词表语义切分...")
        trend_indices, detail_indices = split_full_vocab_by_semantics(
            tokenizer=tokenizer,
            word_embeddings=word_embeddings,
            trend_anchors=None,
            detail_anchors=None,
            verbose=True
        )
        
        # 验证结果
        print("\n[步骤3] 验证切分结果...")
        vocab_size = len(tokenizer)
        
        # 检查不相交
        trend_set = set(trend_indices.cpu().tolist())
        detail_set = set(detail_indices.cpu().tolist())
        overlap = trend_set & detail_set
        
        assert len(overlap) == 0, f"❌ 发现 {len(overlap)} 个重叠词（不应该发生）"
        print("✅ 两个词集完全不相交")
        
        # 检查覆盖
        total = len(trend_set) + len(detail_set)
        assert total == vocab_size, f"❌ 词表覆盖不完整: {total} != {vocab_size}"
        print(f"✅ 词表完全覆盖: {total} = {vocab_size}")
        
        # 检查比例
        trend_ratio = len(trend_indices) / vocab_size
        detail_ratio = len(detail_indices) / vocab_size
        print(f"✅ 趋势桶占比: {trend_ratio*100:.1f}%")
        print(f"✅ 细节桶占比: {detail_ratio*100:.1f}%")
        
        # 打印样本
        print("\n[步骤4] 打印切分结果样本...")
        print_vocab_split_samples(tokenizer, trend_indices, detail_indices, max_print=20)
        
        print("\n✅ vocab_splitter 函数测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ vocab_splitter 函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """测试模型初始化"""
    print("\n" + "=" * 70)
    print("测试 2: TimeLLM 模型初始化（全词表切分模式）")
    print("=" * 70)
    
    try:
        configs = TestConfig()
        
        print("\n[步骤1] 初始化模型...")
        model = Model(configs)
        print("✅ 模型初始化成功")
        
        # 验证全词表切分相关属性
        print("\n[步骤2] 验证全词表切分相关属性...")
        
        assert hasattr(model, 'use_full_vocab_split'), "❌ 缺少 use_full_vocab_split 属性"
        assert model.use_full_vocab_split == True, f"❌ use_full_vocab_split 应为 True，实际为 {model.use_full_vocab_split}"
        print("✅ use_full_vocab_split 属性正确")
        
        assert hasattr(model, 'trend_vocab_embeddings'), "❌ 缺少 trend_vocab_embeddings Buffer"
        assert hasattr(model, 'detail_vocab_embeddings'), "❌ 缺少 detail_vocab_embeddings Buffer"
        print("✅ 切分后的 embeddings Buffer 已注册")
        
        assert hasattr(model, 'trend_mapping'), "❌ 缺少 trend_mapping 层"
        assert hasattr(model, 'detail_mapping'), "❌ 缺少 detail_mapping 层"
        print("✅ 映射层已创建")
        
        # 验证映射层类型（应该是 Linear，不是 MLP）
        assert isinstance(model.trend_mapping, nn.Linear), f"❌ trend_mapping 应为 Linear，实际为 {type(model.trend_mapping)}"
        assert isinstance(model.detail_mapping, nn.Linear), f"❌ detail_mapping 应为 Linear，实际为 {type(model.detail_mapping)}"
        print("✅ 映射层类型正确（Linear，和原版TimeLLM一样）")
        
        # 验证映射层维度
        trend_vocab_size = model.trend_vocab_embeddings.shape[0]
        detail_vocab_size = model.detail_vocab_embeddings.shape[0]
        
        assert model.trend_mapping.in_features == trend_vocab_size, \
            f"❌ trend_mapping 输入维度不匹配: {model.trend_mapping.in_features} != {trend_vocab_size}"
        assert model.trend_mapping.out_features == configs.dual_proto_trend_tokens, \
            f"❌ trend_mapping 输出维度不匹配: {model.trend_mapping.out_features} != {configs.dual_proto_trend_tokens}"
        
        assert model.detail_mapping.in_features == detail_vocab_size, \
            f"❌ detail_mapping 输入维度不匹配: {model.detail_mapping.in_features} != {detail_vocab_size}"
        assert model.detail_mapping.out_features == configs.dual_proto_detail_tokens, \
            f"❌ detail_mapping 输出维度不匹配: {model.detail_mapping.out_features} != {configs.dual_proto_detail_tokens}"
        
        print(f"✅ 映射层维度正确:")
        print(f"   - 趋势映射: Linear({trend_vocab_size:,} → {configs.dual_proto_trend_tokens})")
        print(f"   - 细节映射: Linear({detail_vocab_size:,} → {configs.dual_proto_detail_tokens})")
        
        # 计算参数量
        trend_params = trend_vocab_size * configs.dual_proto_trend_tokens
        detail_params = detail_vocab_size * configs.dual_proto_detail_tokens
        total_params = trend_params + detail_params
        print(f"✅ 参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        print("\n✅ 模型初始化测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 70)
    print("测试 3: 前向传播（全词表切分模式）")
    print("=" * 70)
    
    try:
        configs = TestConfig()
        
        print("\n[步骤1] 初始化模型...")
        model = Model(configs)
        model.eval()
        print("✅ 模型初始化成功")
        
        # 创建测试输入
        print("\n[步骤2] 创建测试输入...")
        batch_size = 2
        seq_len = configs.seq_len
        n_vars = configs.enc_in
        
        x_enc = torch.randn(batch_size, seq_len, n_vars)
        x_mark_enc = torch.zeros(batch_size, seq_len, 4)  # 时间戳特征
        x_dec = torch.randn(batch_size, configs.pred_len, n_vars)
        x_mark_dec = torch.zeros(batch_size, configs.pred_len, 4)
        
        print(f"✅ 输入形状: x_enc {x_enc.shape}, x_mark_enc {x_mark_enc.shape}")
        
        # 前向传播
        print("\n[步骤3] 执行前向传播...")
        with torch.no_grad():
            output = model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 验证输出形状
        print("\n[步骤4] 验证输出形状...")
        expected_shape = (batch_size, configs.pred_len, n_vars)
        assert output.shape == expected_shape, \
            f"❌ 输出形状错误: {output.shape} != {expected_shape}"
        print(f"✅ 输出形状正确: {output.shape}")
        
        # 验证输出不是 NaN 或 Inf
        assert not torch.isnan(output).any(), "❌ 输出包含 NaN"
        assert not torch.isinf(output).any(), "❌ 输出包含 Inf"
        print("✅ 输出值有效（无 NaN/Inf）")
        
        print("\n✅ 前向传播测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prototype_generation():
    """测试原型生成"""
    print("\n" + "=" * 70)
    print("测试 4: 原型生成（全词表切分模式）")
    print("=" * 70)
    
    try:
        configs = TestConfig()
        
        print("\n[步骤1] 初始化模型...")
        model = Model(configs)
        model.eval()
        print("✅ 模型初始化成功")
        
        # 手动生成原型（模拟 forward 中的逻辑）
        print("\n[步骤2] 生成趋势和细节原型...")
        with torch.no_grad():
            trend_prototypes = model.trend_mapping(
                model.trend_vocab_embeddings.permute(1, 0)
            ).permute(1, 0)
            
            detail_prototypes = model.detail_mapping(
                model.detail_vocab_embeddings.permute(1, 0)
            ).permute(1, 0)
        
        # 验证原型形状
        print("\n[步骤3] 验证原型形状...")
        d_llm = model.d_llm
        
        expected_trend_shape = (configs.dual_proto_trend_tokens, d_llm)
        expected_detail_shape = (configs.dual_proto_detail_tokens, d_llm)
        
        assert trend_prototypes.shape == expected_trend_shape, \
            f"❌ 趋势原型形状错误: {trend_prototypes.shape} != {expected_trend_shape}"
        assert detail_prototypes.shape == expected_detail_shape, \
            f"❌ 细节原型形状错误: {detail_prototypes.shape} != {expected_detail_shape}"
        
        print(f"✅ 趋势原型形状: {trend_prototypes.shape}")
        print(f"✅ 细节原型形状: {detail_prototypes.shape}")
        
        # 验证原型值
        assert not torch.isnan(trend_prototypes).any(), "❌ 趋势原型包含 NaN"
        assert not torch.isnan(detail_prototypes).any(), "❌ 细节原型包含 NaN"
        assert not torch.isinf(trend_prototypes).any(), "❌ 趋势原型包含 Inf"
        assert not torch.isinf(detail_prototypes).any(), "❌ 细节原型包含 Inf"
        print("✅ 原型值有效（无 NaN/Inf）")
        
        print("\n✅ 原型生成测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 原型生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mutual_exclusivity():
    """测试互斥性：use_full_vocab_split 和 use_semantic_filtered_mapping 不能同时启用"""
    print("\n" + "=" * 70)
    print("测试 5: 互斥性验证")
    print("=" * 70)
    
    try:
        configs = TestConfig()
        configs.use_full_vocab_split = 1
        configs.use_semantic_filtered_mapping = 1  # 同时启用，应该报错
        
        print("\n[步骤1] 尝试同时启用 use_full_vocab_split 和 use_semantic_filtered_mapping...")
        try:
            model = Model(configs)
            print("❌ 应该抛出 ValueError，但没有抛出")
            return False
        except ValueError as e:
            if "不能同时启用" in str(e):
                print(f"✅ 正确抛出 ValueError: {e}")
                return True
            else:
                print(f"❌ 抛出 ValueError 但消息不正确: {e}")
                return False
        except Exception as e:
            print(f"❌ 抛出意外的异常: {e}")
            return False
        
    except Exception as e:
        print(f"\n❌ 互斥性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 70)
    print("全词表切分功能测试")
    print("=" * 70)
    
    results = []
    
    # 测试 1: vocab_splitter 函数
    results.append(("vocab_splitter 函数", test_vocab_splitter()))
    
    # 测试 2: 模型初始化
    results.append(("模型初始化", test_model_initialization()))
    
    # 测试 3: 前向传播
    results.append(("前向传播", test_forward_pass()))
    
    # 测试 4: 原型生成
    results.append(("原型生成", test_prototype_generation()))
    
    # 测试 5: 互斥性
    results.append(("互斥性验证", test_mutual_exclusivity()))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 70)
    print(f"总计: {passed} 个通过, {failed} 个失败")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return True
    else:
        print(f"\n⚠️  有 {failed} 个测试失败")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

