#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通道拼接融合测试脚本

测试内容：
1. DualReprogrammingLayer 的 channel_concat 融合方法
2. 序列长度保持不变验证
3. 特征维度拼接和投影验证
4. 输出头参数量验证（不应翻倍）
5. 端到端测试
6. 与其他融合方法对比
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
        self.dual_proto_num_tokens = 1000
        self.dual_proto_fusion_method = 'channel_concat'
        self.use_full_vocab_split = 1
        
        # Prompt配置
        self.prompt_domain = 0
        self.content = 'Test dataset description'
        
        # 输出头配置
        self.use_dual_scale_head = 0
        self.use_freq_decoupled_head = 0
        
        # 其他配置
        self.use_cwpr = 0


def test_channel_concat_fusion_basic():
    """测试1: 通道拼接基本功能"""
    print("=" * 70)
    print("测试1: 通道拼接基本功能")
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
        fusion_method='channel_concat',
        gate_bias_init=0.0
    )
    
    print(f"\n融合方法: {layer.fusion_method}")
    
    if layer.fusion_method != 'channel_concat':
        print(f"❌ 融合方法不正确: {layer.fusion_method} != channel_concat")
        return False
    print("✅ 融合方法正确: channel_concat")
    
    # 检查投影层是否存在
    if layer.fusion_projection is None:
        print("❌ 投影层未创建")
        return False
    print("✅ 投影层已创建")
    
    # 检查投影层维度
    expected_in_features = 2 * d_llm
    expected_out_features = d_llm
    if layer.fusion_projection.in_features != expected_in_features:
        print(f"❌ 投影层输入维度不正确: {layer.fusion_projection.in_features} != {expected_in_features}")
        return False
    if layer.fusion_projection.out_features != expected_out_features:
        print(f"❌ 投影层输出维度不正确: {layer.fusion_projection.out_features} != {expected_out_features}")
        return False
    print(f"✅ 投影层维度正确: Linear({expected_in_features}, {expected_out_features})")
    
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
    
    # 验证输出形状：序列长度应该保持不变
    if output.shape != (batch_size, seq_len, d_llm):
        print(f"❌ 输出形状不正确: {output.shape} != ({batch_size}, {seq_len}, {d_llm})")
        return False
    print("✅ 输出形状正确（序列长度保持不变）")
    
    # 验证输出值
    if torch.isnan(output).any():
        print("❌ 输出包含 NaN")
        return False
    print("✅ 输出值合理（无NaN）")
    
    if torch.isinf(output).any():
        print("❌ 输出包含 Inf")
        return False
    print("✅ 输出值合理（无Inf）")
    
    return True


def test_channel_concat_vs_interleave():
    """测试2: 对比 channel_concat 和 interleave 的序列长度"""
    print("\n" + "=" * 70)
    print("测试2: 对比 channel_concat 和 interleave 的序列长度")
    print("=" * 70)
    
    d_model = 16
    d_llm = 768
    n_heads = 4
    batch_size = 1
    seq_len = 5
    num_prototypes = 100
    
    # 创建两个层：channel_concat 和 interleave
    layer_concat = DualReprogrammingLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_keys=d_model // n_heads,
        d_llm=d_llm,
        attention_dropout=0.1,
        fusion_method='channel_concat',
    )
    
    layer_interleave = DualReprogrammingLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_keys=d_model // n_heads,
        d_llm=d_llm,
        attention_dropout=0.1,
        fusion_method='interleave',
    )
    
    # 创建相同的测试输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model)
    detail_embedding = torch.randn(batch_size, seq_len, d_model)
    trend_prototypes = torch.randn(num_prototypes, d_llm)
    detail_prototypes = torch.randn(num_prototypes, d_llm)
    
    layer_concat.eval()
    layer_interleave.eval()
    with torch.no_grad():
        output_concat = layer_concat(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
        output_interleave = layer_interleave(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
    
    print(f"\nchannel_concat 输出形状: {output_concat.shape}")
    print(f"interleave 输出形状: {output_interleave.shape}")
    
    # 验证 channel_concat 保持序列长度
    if output_concat.shape[1] != seq_len:
        print(f"❌ channel_concat 序列长度不正确: {output_concat.shape[1]} != {seq_len}")
        return False
    print(f"✅ channel_concat 序列长度正确: {output_concat.shape[1]} == {seq_len}")
    
    # 验证 interleave 序列长度翻倍
    if output_interleave.shape[1] != 2 * seq_len:
        print(f"❌ interleave 序列长度不正确: {output_interleave.shape[1]} != {2 * seq_len}")
        return False
    print(f"✅ interleave 序列长度正确: {output_interleave.shape[1]} == {2 * seq_len}")
    
    # 验证特征维度
    if output_concat.shape[2] != d_llm:
        print(f"❌ channel_concat 特征维度不正确: {output_concat.shape[2]} != {d_llm}")
        return False
    print(f"✅ channel_concat 特征维度正确: {output_concat.shape[2]} == {d_llm}")
    
    if output_interleave.shape[2] != d_llm:
        print(f"❌ interleave 特征维度不正确: {output_interleave.shape[2]} != {d_llm}")
        return False
    print(f"✅ interleave 特征维度正确: {output_interleave.shape[2]} == {d_llm}")
    
    return True


def test_channel_concat_projection():
    """测试3: 验证投影层的功能"""
    print("\n" + "=" * 70)
    print("测试3: 验证投影层的功能")
    print("=" * 70)
    
    d_model = 16
    d_llm = 768
    n_heads = 4
    batch_size = 2
    seq_len = 10
    num_prototypes = 100
    
    layer = DualReprogrammingLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_keys=d_model // n_heads,
        d_llm=d_llm,
        attention_dropout=0.1,
        fusion_method='channel_concat',
    )
    
    # 创建测试输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model)
    detail_embedding = torch.randn(batch_size, seq_len, d_model)
    trend_prototypes = torch.randn(num_prototypes, d_llm)
    detail_prototypes = torch.randn(num_prototypes, d_llm)
    
    layer.eval()
    with torch.no_grad():
        # 手动执行前两步，验证拼接
        sem_trend = layer.trend_reprogramming(trend_embedding, trend_prototypes, trend_prototypes)
        sem_detail = layer.detail_reprogramming(detail_embedding, detail_prototypes, detail_prototypes)
        
        # 手动拼接
        concat_output = torch.cat([sem_trend, sem_detail], dim=-1)
        
        # 通过投影层
        projected_output = layer.fusion_projection(concat_output)
        
        # 完整前向传播
        full_output = layer(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
    
    print(f"\n拼接后形状: {concat_output.shape}")
    print(f"投影后形状: {projected_output.shape}")
    print(f"完整输出形状: {full_output.shape}")
    
    # 验证拼接维度
    if concat_output.shape != (batch_size, seq_len, 2 * d_llm):
        print(f"❌ 拼接后形状不正确: {concat_output.shape} != ({batch_size}, {seq_len}, {2 * d_llm})")
        return False
    print("✅ 拼接后形状正确")
    
    # 验证投影后维度
    if projected_output.shape != (batch_size, seq_len, d_llm):
        print(f"❌ 投影后形状不正确: {projected_output.shape} != ({batch_size}, {seq_len}, {d_llm})")
        return False
    print("✅ 投影后形状正确")
    
    # 验证完整输出与手动计算一致
    if not torch.allclose(projected_output, full_output, atol=1e-5):
        print("❌ 完整输出与手动计算不一致")
        return False
    print("✅ 完整输出与手动计算一致")
    
    return True


def test_channel_concat_head_params():
    """测试4: 验证输出头参数量（不应翻倍）"""
    print("\n" + "=" * 70)
    print("测试4: 验证输出头参数量（不应翻倍）")
    print("=" * 70)
    
    config = TestConfig()
    config.dual_proto_fusion_method = 'channel_concat'
    
    model = Model(config)
    
    # 计算 head_nf
    patch_nums = int((config.seq_len - config.patch_len) / config.stride + 2)
    expected_head_nf = config.d_ff * patch_nums  # 不应翻倍
    
    print(f"\n配置:")
    print(f"  - seq_len: {config.seq_len}")
    print(f"  - patch_len: {config.patch_len}")
    print(f"  - stride: {config.stride}")
    print(f"  - patch_nums: {patch_nums}")
    print(f"  - d_ff: {config.d_ff}")
    print(f"  - fusion_method: {config.dual_proto_fusion_method}")
    
    print(f"\n模型 head_nf: {model.head_nf}")
    print(f"预期 head_nf: {expected_head_nf}")
    
    if model.head_nf != expected_head_nf:
        print(f"❌ head_nf 不正确: {model.head_nf} != {expected_head_nf}")
        return False
    print("✅ head_nf 正确（未翻倍）")
    
    # 计算输出头参数量
    # FlattenHead 的参数量计算：
    # - 输入形状: (B, n_vars, d_ff, patch_nums)
    # - flatten 后: (B, n_vars, d_ff * patch_nums) = (B, n_vars, head_nf)
    # - Linear(head_nf, pred_len): weight = head_nf * pred_len, bias = pred_len
    # - 总参数量: head_nf * pred_len + pred_len (共享的 Linear 层，所有变量共用)
    if hasattr(model.output_projection, 'linear'):
        head_params = sum(p.numel() for p in model.output_projection.linear.parameters())
        print(f"\n输出头参数量: {head_params:,}")
        
        # 计算预期参数量：FlattenHead 使用共享的 Linear 层
        # Linear(head_nf, pred_len) 的参数量 = head_nf * pred_len + pred_len (bias)
        expected_params = expected_head_nf * config.pred_len + config.pred_len
        print(f"预期参数量: {expected_params:,}")
        print(f"  - weight: {expected_head_nf * config.pred_len:,}")
        print(f"  - bias: {config.pred_len:,}")
        
        if head_params != expected_params:
            print(f"❌ 输出头参数量不正确: {head_params} != {expected_params}")
            print(f"   差异: {abs(head_params - expected_params):,}")
            return False
        print("✅ 输出头参数量正确（未翻倍）")
    
    return True


def test_channel_concat_end_to_end():
    """测试5: 端到端测试"""
    print("\n" + "=" * 70)
    print("测试5: 端到端测试")
    print("=" * 70)
    
    config = TestConfig()
    config.dual_proto_fusion_method = 'channel_concat'
    
    model = Model(config)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # 假设4个时间特征
    x_dec = torch.randn(batch_size, config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, config.pred_len, 4)
    
    print(f"\n输入形状:")
    print(f"  - x_enc: {x_enc.shape}")
    print(f"  - x_mark_enc: {x_mark_enc.shape}")
    print(f"  - x_dec: {x_dec.shape}")
    print(f"  - x_mark_dec: {x_mark_dec.shape}")
    
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\n输出形状: {output.shape}")
    print(f"预期形状: ({batch_size}, {config.pred_len}, {config.c_out})")
    
    if output.shape != (batch_size, config.pred_len, config.c_out):
        print(f"❌ 输出形状不正确: {output.shape} != ({batch_size}, {config.pred_len}, {config.c_out})")
        return False
    print("✅ 输出形状正确")
    
    # 验证输出值
    if torch.isnan(output).any():
        print("❌ 输出包含 NaN")
        return False
    print("✅ 输出值合理（无NaN）")
    
    if torch.isinf(output).any():
        print("❌ 输出包含 Inf")
        return False
    print("✅ 输出值合理（无Inf）")
    
    return True


def test_channel_concat_vs_mean():
    """测试6: 对比 channel_concat 和 mean 融合"""
    print("\n" + "=" * 70)
    print("测试6: 对比 channel_concat 和 mean 融合")
    print("=" * 70)
    
    d_model = 16
    d_llm = 768
    n_heads = 4
    batch_size = 2
    seq_len = 10
    num_prototypes = 100
    
    # 创建两个层
    layer_concat = DualReprogrammingLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_keys=d_model // n_heads,
        d_llm=d_llm,
        attention_dropout=0.1,
        fusion_method='channel_concat',
    )
    
    layer_mean = DualReprogrammingLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_keys=d_model // n_heads,
        d_llm=d_llm,
        attention_dropout=0.1,
        fusion_method='mean',
    )
    
    # 创建相同的测试输入
    trend_embedding = torch.randn(batch_size, seq_len, d_model)
    detail_embedding = torch.randn(batch_size, seq_len, d_model)
    trend_prototypes = torch.randn(num_prototypes, d_llm)
    detail_prototypes = torch.randn(num_prototypes, d_llm)
    
    layer_concat.eval()
    layer_mean.eval()
    with torch.no_grad():
        output_concat = layer_concat(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
        output_mean = layer_mean(trend_embedding, detail_embedding, trend_prototypes, detail_prototypes)
    
    print(f"\nchannel_concat 输出形状: {output_concat.shape}")
    print(f"mean 输出形状: {output_mean.shape}")
    
    # 验证形状相同
    if output_concat.shape != output_mean.shape:
        print(f"❌ 输出形状不同: {output_concat.shape} != {output_mean.shape}")
        return False
    print("✅ 输出形状相同")
    
    # 验证输出不同（因为融合方式不同）
    if torch.allclose(output_concat, output_mean, atol=1e-3):
        print("⚠️  警告: 两种融合方法输出过于接近（可能有问题）")
    else:
        print("✅ 两种融合方法输出不同（符合预期）")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Channel Concatenation 融合方法测试套件")
    print("=" * 70)
    
    tests = [
        ("基本功能测试", test_channel_concat_fusion_basic),
        ("序列长度对比测试", test_channel_concat_vs_interleave),
        ("投影层功能测试", test_channel_concat_projection),
        ("输出头参数量测试", test_channel_concat_head_params),
        ("端到端测试", test_channel_concat_end_to_end),
        ("融合方法对比测试", test_channel_concat_vs_mean),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 执行失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {len(results)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit(main())

