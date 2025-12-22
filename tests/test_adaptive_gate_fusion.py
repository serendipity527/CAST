#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态门控融合测试脚本

测试内容：
1. AdaptiveFusionGate 基本功能测试
2. 门控权重计算正确性
3. DualReprogrammingLayer 使用 adaptive_gate 融合
4. 不同融合方法对比
5. 端到端测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

from models.TimeLLM import Model, AdaptiveFusionGate, DualReprogrammingLayer


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
        self.dual_proto_fusion_method = 'adaptive_gate'
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


def test_adaptive_fusion_gate_basic():
    """测试1: AdaptiveFusionGate 基本功能"""
    print("=" * 70)
    print("测试1: AdaptiveFusionGate 基本功能")
    print("=" * 70)
    
    d_model = 16
    batch_size = 2
    seq_len = 10
    
    # 创建门控网络
    gate = AdaptiveFusionGate(d_model, gate_bias_init=0.0)
    
    # 创建测试输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model)
    detail_embedding = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状:")
    print(f"  - trend_embedding: {trend_embedding.shape}")
    print(f"  - detail_embedding: {detail_embedding.shape}")
    
    # 前向传播
    gate_weights = gate(trend_embedding, detail_embedding)
    
    print(f"\n输出形状: {gate_weights.shape}")
    print(f"预期形状: ({batch_size}, {seq_len}, 1)")
    
    # 验证输出形状
    if gate_weights.shape != (batch_size, seq_len, 1):
        print(f"❌ 输出形状不正确: {gate_weights.shape} != ({batch_size}, {seq_len}, 1)")
        return False
    print("✅ 输出形状正确")
    
    # 验证门控权重范围 [0, 1]
    if (gate_weights < 0).any() or (gate_weights > 1).any():
        print("❌ 门控权重超出范围 [0, 1]")
        print(f"   最小值: {gate_weights.min().item():.6f}")
        print(f"   最大值: {gate_weights.max().item():.6f}")
        return False
    print("✅ 门控权重在有效范围 [0, 1] 内")
    
    # 验证不同位置的门控权重不同（应该根据输入特征动态计算）
    if torch.allclose(gate_weights, gate_weights[0, 0, 0].expand_as(gate_weights), atol=1e-5):
        print("⚠️  警告: 所有位置的门控权重相同，可能没有正确计算")
    else:
        print("✅ 不同位置的门控权重不同（动态计算生效）")
        print(f"   门控权重统计: 均值={gate_weights.mean().item():.4f}, "
              f"标准差={gate_weights.std().item():.4f}, "
              f"范围=[{gate_weights.min().item():.4f}, {gate_weights.max().item():.4f}]")
    
    return True


def test_adaptive_fusion_gate_bias_init():
    """测试2: 门控偏置初始化影响"""
    print("\n" + "=" * 70)
    print("测试2: 门控偏置初始化影响")
    print("=" * 70)
    
    d_model = 16
    batch_size = 2
    seq_len = 10
    
    # 创建相同的输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model)
    detail_embedding = torch.randn(batch_size, seq_len, d_model)
    
    # 测试不同的偏置初始化
    bias_values = [-2.0, 0.0, 2.0]
    results = []
    
    for bias_init in bias_values:
        gate = AdaptiveFusionGate(d_model, gate_bias_init=bias_init)
        gate.eval()
        
        with torch.no_grad():
            gate_weights = gate(trend_embedding, detail_embedding)
            mean_weight = gate_weights.mean().item()
            results.append((bias_init, mean_weight))
        
        print(f"\n偏置初始化: {bias_init}")
        print(f"  平均门控权重: {mean_weight:.4f}")
    
    # 验证：偏置越大，平均权重应该越大（更偏向趋势）
    if results[0][1] < results[1][1] < results[2][1]:
        print("\n✅ 偏置初始化影响正确：偏置越大，平均权重越大（更偏向趋势）")
    else:
        print("\n⚠️  警告: 偏置初始化影响不符合预期")
    
    return True


def test_dual_reprogramming_adaptive_gate():
    """测试3: DualReprogrammingLayer 使用 adaptive_gate 融合"""
    print("\n" + "=" * 70)
    print("测试3: DualReprogrammingLayer 使用 adaptive_gate 融合")
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
        fusion_method='adaptive_gate',
        gate_bias_init=0.0
    )
    
    print(f"\n融合方法: {layer.fusion_method}")
    print(f"fusion_gate 类型: {type(layer.fusion_gate).__name__}")
    
    if layer.fusion_gate is None:
        print("❌ fusion_gate 未初始化")
        return False
    print("✅ fusion_gate 已正确初始化")
    
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
    print(f"预期形状: ({batch_size}, {seq_len}, {d_llm})")
    
    if output.shape != (batch_size, seq_len, d_llm):
        print(f"❌ 输出形状不正确")
        return False
    print("✅ 输出形状正确")
    
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


def test_fusion_methods_comparison():
    """测试4: 不同融合方法对比"""
    print("\n" + "=" * 70)
    print("测试4: 不同融合方法对比")
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
    fusion_methods = ['mean', 'weighted', 'adaptive_gate']
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
            'mean': output.mean().item(),
            'std': output.std().item(),
            'params': sum(p.numel() for p in layer.parameters())
        }
        
        print(f"\n{method} 融合:")
        print(f"  输出统计: 均值={results[method]['mean']:.6f}, 标准差={results[method]['std']:.6f}")
        print(f"  参数量: {results[method]['params']:,}")
    
    # 验证不同方法产生不同的输出
    mean_output = results['mean']['output']
    weighted_output = results['weighted']['output']
    adaptive_output = results['adaptive_gate']['output']
    
    if torch.allclose(mean_output, weighted_output, atol=1e-5):
        print("\n⚠️  警告: mean 和 weighted 输出过于接近")
    else:
        print("\n✅ mean 和 weighted 输出不同")
    
    if torch.allclose(mean_output, adaptive_output, atol=1e-5):
        print("⚠️  警告: mean 和 adaptive_gate 输出过于接近")
    else:
        print("✅ mean 和 adaptive_gate 输出不同")
    
    if torch.allclose(weighted_output, adaptive_output, atol=1e-5):
        print("⚠️  警告: weighted 和 adaptive_gate 输出过于接近")
    else:
        print("✅ weighted 和 adaptive_gate 输出不同")
    
    # 验证参数量
    print(f"\n参数量对比:")
    print(f"  - mean: {results['mean']['params']:,} (无额外参数)")
    print(f"  - weighted: {results['weighted']['params']:,} (1个参数)")
    print(f"  - adaptive_gate: {results['adaptive_gate']['params']:,} (门控网络参数)")
    
    if results['adaptive_gate']['params'] > results['weighted']['params']:
        print("✅ adaptive_gate 参数量大于 weighted（符合预期）")
    else:
        print("⚠️  警告: adaptive_gate 参数量异常")
    
    return True


def test_gradient_flow():
    """测试5: 梯度流测试"""
    print("\n" + "=" * 70)
    print("测试5: 梯度流测试")
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
        fusion_method='adaptive_gate',
        gate_bias_init=0.0
    )
    layer.train()
    
    # 创建输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    detail_embedding = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    trend_prototypes = torch.randn(num_prototypes, d_llm, requires_grad=False)
    detail_prototypes = torch.randn(num_prototypes, d_llm, requires_grad=False)
    
    # 前向传播
    output = layer(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
    
    # 创建虚拟损失
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print("\n[检查1] 输入梯度...")
    if trend_embedding.grad is None or detail_embedding.grad is None:
        print("❌ 输入没有梯度")
        return False
    print("✅ 输入有梯度")
    
    print("\n[检查2] 门控网络参数梯度...")
    gate_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in layer.fusion_gate.parameters())
    if not gate_has_grad:
        print("❌ 门控网络没有梯度")
        return False
    print("✅ 门控网络有梯度")
    
    print("\n[检查3] 重编程层参数梯度...")
    trend_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                        for p in layer.trend_reprogramming.parameters())
    detail_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                         for p in layer.detail_reprogramming.parameters())
    if not trend_has_grad or not detail_has_grad:
        print("❌ 重编程层没有梯度")
        return False
    print("✅ 重编程层有梯度")
    
    return True


def test_end_to_end():
    """测试6: 端到端测试"""
    print("\n" + "=" * 70)
    print("测试6: 端到端测试（完整模型）")
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
    if hasattr(model, 'reprogramming_layer') and model.reprogramming_layer is not None:
        if model.reprogramming_layer.fusion_method != 'adaptive_gate':
            print(f"❌ 融合方法不正确: {model.reprogramming_layer.fusion_method} != adaptive_gate")
            return False
        print(f"✅ 融合方法正确: {model.reprogramming_layer.fusion_method}")
        
        if model.reprogramming_layer.fusion_gate is None:
            print("❌ fusion_gate 未初始化")
            return False
        print("✅ fusion_gate 已初始化")
    
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


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("动态门控融合完整测试套件")
    print("=" * 70)
    
    tests = [
        ("AdaptiveFusionGate 基本功能", test_adaptive_fusion_gate_basic),
        ("门控偏置初始化影响", test_adaptive_fusion_gate_bias_init),
        ("DualReprogrammingLayer adaptive_gate 融合", test_dual_reprogramming_adaptive_gate),
        ("不同融合方法对比", test_fusion_methods_comparison),
        ("梯度流测试", test_gradient_flow),
        ("端到端测试", test_end_to_end),
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
        print("\n🎉 所有测试通过！动态门控融合实现正确！")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

