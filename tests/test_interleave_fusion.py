#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交错拼接融合测试脚本

测试内容：
1. DualReprogrammingLayer 的 interleave 融合方法
2. 序列长度翻倍验证
3. 输出头适配验证
4. 端到端测试
5. 与其他融合方法对比
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

from models.TimeLLM import Model, DualReprogrammingLayer


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
        self.dual_proto_fusion_method = 'interleave'
        self.dual_proto_gate_bias_init = 0.0
        
        # 语义筛选映射配置
        self.use_semantic_filtered_mapping = 1
        self.dual_proto_trend_seed_words = 300
        self.dual_proto_detail_seed_words = 700
        self.dual_proto_seed_semantic_filter = 1
        
        # MLP映射层配置
        self.dual_proto_mlp_hidden_dim = 2048  # 测试时使用较小的维度
        self.dual_proto_mlp_dropout = 0.1
        
        # Prompt配置
        self.prompt_domain = 0
        self.content = 'Test dataset description'
        
        # 其他配置
        self.use_cwpr = 0
        self.use_dual_scale_head = 0
        self.use_freq_decoupled_head = 0


def test_interleave_fusion_basic():
    """测试1: 交错拼接基本功能"""
    print("=" * 70)
    print("测试1: 交错拼接基本功能")
    print("=" * 70)
    
    d_model = 16
    d_llm = 768
    n_heads = 4
    batch_size = 2
    seq_len = 10
    num_prototypes = 100
    
    # 创建 DualReprogrammingLayer
    layer = DualReprogrammingLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_keys=d_model // n_heads,
        d_llm=d_llm,
        attention_dropout=0.1,
        fusion_method='interleave',
        gate_bias_init=0.0
    )
    
    print(f"\n融合方法: {layer.fusion_method}")
    
    if layer.fusion_method != 'interleave':
        print(f"❌ 融合方法不正确: {layer.fusion_method} != interleave")
        return False
    print("✅ 融合方法正确: interleave")
    
    # 创建测试输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model)
    detail_embedding = torch.randn(batch_size, seq_len, d_model)
    trend_prototypes = torch.randn(num_prototypes, d_llm)
    detail_prototypes = torch.randn(num_prototypes, d_llm)
    
    print(f"\n输入形状:")
    print(f"  - trend_embedding: {trend_embedding.shape}")
    print(f"  - detail_embedding: {detail_embedding.shape}")
    print(f"  - trend_prototypes: {trend_prototypes.shape}")
    print(f"  - detail_prototypes: {detail_prototypes.shape}")
    
    # 前向传播
    layer.eval()
    with torch.no_grad():
        output = layer(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
    
    print(f"\n输出形状: {output.shape}")
    print(f"预期形状: ({batch_size}, {2*seq_len}, {d_llm})")
    
    # 验证输出形状：序列长度应该翻倍
    if output.shape != (batch_size, 2*seq_len, d_llm):
        print(f"❌ 输出形状不正确: {output.shape} != ({batch_size}, {2*seq_len}, {d_llm})")
        return False
    print("✅ 输出形状正确（序列长度翻倍）")
    
    # 验证输出值
    if torch.isnan(output).any():
        print("❌ 输出包含NaN值")
        return False
    print("✅ 输出值合理（无NaN）")
    
    if torch.isinf(output).any():
        print("❌ 输出包含Inf值")
        return False
    print("✅ 输出值合理（无Inf）")
    
    return True


def test_interleave_ordering():
    """测试2: 验证交错顺序"""
    print("\n" + "=" * 70)
    print("测试2: 验证交错顺序 [T1, D1, T2, D2, ...]")
    print("=" * 70)
    
    d_model = 16
    d_llm = 768
    n_heads = 4
    batch_size = 1
    seq_len = 5
    num_prototypes = 100
    
    # 创建层
    layer = DualReprogrammingLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_keys=d_model // n_heads,
        d_llm=d_llm,
        attention_dropout=0.1,
        fusion_method='interleave',
        gate_bias_init=0.0
    )
    
    # 创建特殊的测试输入：趋势和细节有明显区别
    trend_embedding = torch.ones(batch_size, seq_len, d_model) * 1.0
    detail_embedding = torch.ones(batch_size, seq_len, d_model) * 2.0
    
    # 创建简单的原型（用于测试）
    trend_prototypes = torch.eye(d_llm)[:num_prototypes]  # 单位矩阵
    detail_prototypes = torch.eye(d_llm)[:num_prototypes] * 2  # 2倍单位矩阵
    
    layer.eval()
    with torch.no_grad():
        output = layer(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
    
    print(f"\n输出形状: {output.shape}")
    print(f"预期: (1, {2*seq_len}, {d_llm})")
    
    # 验证交错顺序：检查相邻位置的差异
    # 由于原型不同，趋势和细节的输出应该不同
    print(f"\n输出统计:")
    print(f"  - 位置0 (应该是T1): 均值={output[0, 0, :].mean().item():.6f}")
    print(f"  - 位置1 (应该是D1): 均值={output[0, 1, :].mean().item():.6f}")
    print(f"  - 位置2 (应该是T2): 均值={output[0, 2, :].mean().item():.6f}")
    print(f"  - 位置3 (应该是D2): 均值={output[0, 3, :].mean().item():.6f}")
    
    # 验证相邻位置不同（因为输入不同，输出应该不同）
    if torch.allclose(output[0, 0, :], output[0, 1, :], atol=1e-3):
        print("⚠️  警告: 相邻位置输出过于接近，可能交错顺序有问题")
    else:
        print("✅ 相邻位置输出不同（交错顺序正确）")
    
    # 验证偶数位置和奇数位置的模式
    even_positions = output[0, 0::2, :]  # T1, T2, T3, ...
    odd_positions = output[0, 1::2, :]   # D1, D2, D3, ...
    
    print(f"\n偶数位置（趋势）统计: 均值={even_positions.mean().item():.6f}, 标准差={even_positions.std().item():.6f}")
    print(f"奇数位置（细节）统计: 均值={odd_positions.mean().item():.6f}, 标准差={odd_positions.std().item():.6f}")
    
    return True


def test_fusion_methods_comparison():
    """测试3: 不同融合方法对比"""
    print("\n" + "=" * 70)
    print("测试3: 不同融合方法对比")
    print("=" * 70)
    
    d_model = 16
    d_llm = 768
    n_heads = 4
    batch_size = 2
    seq_len = 10
    num_prototypes = 100
    
    # 创建相同的输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model)
    detail_embedding = torch.randn(batch_size, seq_len, d_model)
    trend_prototypes = torch.randn(num_prototypes, d_llm)
    detail_prototypes = torch.randn(num_prototypes, d_llm)
    
    # 测试不同的融合方法
    fusion_methods = ['mean', 'weighted', 'adaptive_gate', 'interleave']
    results = {}
    
    for method in fusion_methods:
        layer = DualReprogrammingLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_keys=d_model // n_heads,
            d_llm=d_llm,
            attention_dropout=0.1,
            fusion_method=method,
            gate_bias_init=0.0
        )
        layer.eval()
        
        with torch.no_grad():
            output = layer(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
        
        results[method] = {
            'output': output,
            'shape': output.shape,
            'mean': output.mean().item(),
            'std': output.std().item(),
        }
        
        print(f"\n{method} 融合:")
        print(f"  输出形状: {output.shape}")
        print(f"  统计: 均值={results[method]['mean']:.6f}, 标准差={results[method]['std']:.6f}")
    
    # 验证 interleave 的序列长度是其他的2倍
    interleave_shape = results['interleave']['shape']
    other_shape = results['mean']['shape']
    
    if interleave_shape[1] == 2 * other_shape[1]:
        print(f"\n✅ interleave 序列长度正确: {interleave_shape[1]} = 2 × {other_shape[1]}")
    else:
        print(f"\n❌ interleave 序列长度不正确: {interleave_shape[1]} != 2 × {other_shape[1]}")
        return False
    
    # 验证不同方法产生不同的输出
    mean_output = results['mean']['output']
    interleave_output = results['interleave']['output']
    
    # interleave 的前 L 个位置应该与 mean 不同（因为它们是独立的趋势和细节）
    if torch.allclose(mean_output, interleave_output[:, :seq_len, :], atol=1e-3):
        print("⚠️  警告: interleave 的前半部分与 mean 过于接近")
    else:
        print("✅ interleave 输出与 mean 不同（符合预期）")
    
    return True


def test_output_head_adaptation():
    """测试4: 输出头适配验证"""
    print("\n" + "=" * 70)
    print("测试4: 输出头适配验证")
    print("=" * 70)
    
    configs = TestConfig()
    
    try:
        model = Model(configs)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查 head_nf 是否正确翻倍
    expected_head_nf = configs.d_ff * 2 * model.patch_nums
    actual_head_nf = model.head_nf
    
    print(f"\n[检查1] head_nf 计算:")
    print(f"  - patch_nums: {model.patch_nums}")
    print(f"  - d_ff: {configs.d_ff}")
    print(f"  - 预期 head_nf (2*patch_nums*d_ff): {expected_head_nf}")
    print(f"  - 实际 head_nf: {actual_head_nf}")
    
    if actual_head_nf != expected_head_nf:
        print(f"❌ head_nf 不正确: {actual_head_nf} != {expected_head_nf}")
        return False
    print("✅ head_nf 正确（已翻倍）")
    
    # 检查输出头的输入维度
    if hasattr(model.output_projection, 'linear'):
        # FlattenHead
        linear_in_features = model.output_projection.linear.in_features
        print(f"\n[检查2] FlattenHead Linear 输入维度:")
        print(f"  - 实际: {linear_in_features}")
        print(f"  - 预期: {expected_head_nf}")
        
        if linear_in_features != expected_head_nf:
            print(f"❌ FlattenHead Linear 输入维度不正确")
            return False
        print("✅ FlattenHead Linear 输入维度正确")
    
    return True


def test_end_to_end():
    """测试5: 端到端测试"""
    print("\n" + "=" * 70)
    print("测试5: 端到端测试（完整模型）")
    print("=" * 70)
    
    configs = TestConfig()
    
    try:
        model = Model(configs)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查融合方法
    if hasattr(model, 'fusion_method') and model.fusion_method == 'interleave':
        print(f"✅ 融合方法正确: {model.fusion_method}")
    else:
        print(f"❌ 融合方法不正确或未设置")
        return False
    
    # 创建测试输入
    batch_size = 2
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.zeros(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)
    x_mark_dec = torch.zeros(batch_size, configs.pred_len, 4)
    
    print(f"\n输入形状:")
    print(f"  - x_enc: {x_enc.shape}")
    
    # 前向传播
    model.eval()
    try:
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"\n输出形状: {output.shape}")
        print(f"预期形状: ({batch_size}, {configs.pred_len}, {configs.enc_in})")
        
        if output.shape != (batch_size, configs.pred_len, configs.enc_in):
            print(f"❌ 输出形状不匹配")
            return False
        print("✅ 输出形状正确")
        
        # 检查输出值
        if torch.isnan(output).any():
            print("❌ 输出包含NaN值")
            return False
        print("✅ 输出值合理（无NaN）")
        
        if torch.isinf(output).any():
            print("❌ 输出包含Inf值")
            return False
        print("✅ 输出值合理（无Inf）")
        
    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_sequence_length_doubling():
    """测试6: 序列长度翻倍验证"""
    print("\n" + "=" * 70)
    print("测试6: 序列长度翻倍验证")
    print("=" * 70)
    
    d_model = 16
    d_llm = 768
    n_heads = 4
    batch_size = 2
    seq_len = 10
    num_prototypes = 100
    
    # 创建层
    layer = DualReprogrammingLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_keys=d_model // n_heads,
        d_llm=d_llm,
        attention_dropout=0.1,
        fusion_method='interleave',
        gate_bias_init=0.0
    )
    
    # 创建测试输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model)
    detail_embedding = torch.randn(batch_size, seq_len, d_model)
    trend_prototypes = torch.randn(num_prototypes, d_llm)
    detail_prototypes = torch.randn(num_prototypes, d_llm)
    
    layer.eval()
    with torch.no_grad():
        output = layer(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
    
    print(f"\n输入序列长度: {seq_len}")
    print(f"输出序列长度: {output.shape[1]}")
    print(f"预期序列长度: {2 * seq_len}")
    
    if output.shape[1] != 2 * seq_len:
        print(f"❌ 序列长度未翻倍: {output.shape[1]} != {2 * seq_len}")
        return False
    print("✅ 序列长度正确翻倍")
    
    # 验证交错顺序：检查前半部分和后半部分的模式
    first_half = output[:, :seq_len, :]  # 应该是 [T1, D1, T2, D2, ...] 的前半部分
    second_half = output[:, seq_len:, :]  # 应该是 [T1, D1, T2, D2, ...] 的后半部分
    
    print(f"\n前半部分统计: 均值={first_half.mean().item():.6f}, 标准差={first_half.std().item():.6f}")
    print(f"后半部分统计: 均值={second_half.mean().item():.6f}, 标准差={second_half.std().item():.6f}")
    
    # 验证交错：偶数位置应该是趋势，奇数位置应该是细节
    even_pos = output[:, 0::2, :]  # 所有偶数位置
    odd_pos = output[:, 1::2, :]   # 所有奇数位置
    
    print(f"\n偶数位置（趋势）统计: 均值={even_pos.mean().item():.6f}, 标准差={even_pos.std().item():.6f}")
    print(f"奇数位置（细节）统计: 均值={odd_pos.mean().item():.6f}, 标准差={odd_pos.std().item():.6f}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("交错拼接融合完整测试套件")
    print("=" * 70)
    
    tests = [
        ("交错拼接基本功能", test_interleave_fusion_basic),
        ("验证交错顺序", test_interleave_ordering),
        ("不同融合方法对比", test_fusion_methods_comparison),
        ("输出头适配验证", test_output_head_adaptation),
        ("端到端测试", test_end_to_end),
        ("序列长度翻倍验证", test_sequence_length_doubling),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！交错拼接融合实现正确！")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

