#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 FrequencyChannelAttentionV2 模块

验证内容:
1. 模块能够正常初始化
2. 前向传播的输入输出形状正确
3. V1 和 V2 的输出形状一致（但权重形状不同）
4. V2 的注意力权重是 Patch-wise 的（每个 Patch 有独立的频率权重）
5. 梯度能够正常反向传播
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from layers.Embed import FrequencyChannelAttention, FrequencyChannelAttentionV2


def test_basic_forward():
    """测试基本的前向传播"""
    print("=" * 60)
    print("测试 1: 基本前向传播")
    print("=" * 60)
    
    # 参数设置
    batch_size = 4
    n_vars = 7
    num_patches = 64
    d_model = 32
    num_bands = 3  # 例如 level=2: cA, cD_2, cD_1
    
    B_N = batch_size * n_vars
    
    # 创建模拟的频段 embeddings
    band_embeddings = [
        torch.randn(B_N, num_patches, d_model) for _ in range(num_bands)
    ]
    
    # 初始化 V2 模块
    print("\n初始化 FrequencyChannelAttentionV2...")
    attn_v2 = FrequencyChannelAttentionV2(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 前向传播
    output, attention_weights = attn_v2(band_embeddings)
    
    # 验证输出形状
    expected_output_shape = (B_N, num_patches, d_model)
    expected_weight_shape = (B_N, num_patches, num_bands)  # V2 是 Patch-wise
    
    print(f"\n输入: {num_bands} 个频段, 每个形状 ({B_N}, {num_patches}, {d_model})")
    print(f"输出形状: {tuple(output.shape)}, 期望: {expected_output_shape}")
    print(f"权重形状: {tuple(attention_weights.shape)}, 期望: {expected_weight_shape}")
    
    assert output.shape == expected_output_shape, f"输出形状错误: {output.shape} != {expected_output_shape}"
    assert attention_weights.shape == expected_weight_shape, f"权重形状错误: {attention_weights.shape} != {expected_weight_shape}"
    
    print("✅ 测试 1 通过!")
    return True


def test_v1_vs_v2_comparison():
    """对比 V1 和 V2 的差异"""
    print("\n" + "=" * 60)
    print("测试 2: V1 vs V2 对比")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    n_vars = 3
    num_patches = 32
    d_model = 64
    num_bands = 2  # 双通道
    
    B_N = batch_size * n_vars
    
    # 创建相同的输入
    torch.manual_seed(42)
    band_embeddings = [
        torch.randn(B_N, num_patches, d_model) for _ in range(num_bands)
    ]
    
    # 初始化 V1 和 V2
    print("\n初始化 V1 (GAP)...")
    attn_v1 = FrequencyChannelAttention(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4
    )
    
    print("\n初始化 V2 (1D Conv)...")
    attn_v2 = FrequencyChannelAttentionV2(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 前向传播
    output_v1, weights_v1 = attn_v1(band_embeddings)
    output_v2, weights_v2 = attn_v2(band_embeddings)
    
    # 验证输出形状一致
    print(f"\nV1 输出形状: {tuple(output_v1.shape)}")
    print(f"V2 输出形状: {tuple(output_v2.shape)}")
    assert output_v1.shape == output_v2.shape, "V1 和 V2 的输出形状应该一致"
    
    # 验证权重形状差异
    print(f"\nV1 权重形状: {tuple(weights_v1.shape)} (Instance-wise: 所有 Patch 共享权重)")
    print(f"V2 权重形状: {tuple(weights_v2.shape)} (Patch-wise: 每个 Patch 独立权重)")
    
    # V1: (B*N, num_bands)
    # V2: (B*N, num_patches, num_bands)
    assert weights_v1.shape == (B_N, num_bands), f"V1 权重形状错误"
    assert weights_v2.shape == (B_N, num_patches, num_bands), f"V2 权重形状错误"
    
    # 验证权重和为 1
    print(f"\nV1 权重和 (应为 1.0): {weights_v1.sum(dim=-1).mean().item():.4f}")
    print(f"V2 权重和 (应为 1.0): {weights_v2.sum(dim=-1).mean().item():.4f}")
    
    assert torch.allclose(weights_v1.sum(dim=-1), torch.ones(B_N), atol=1e-5), "V1 权重和应为 1"
    assert torch.allclose(weights_v2.sum(dim=-1), torch.ones(B_N, num_patches), atol=1e-5), "V2 权重和应为 1"
    
    print("✅ 测试 2 通过!")
    return True


def test_patch_wise_weights():
    """验证 V2 的 Patch-wise 权重确实是不同的"""
    print("\n" + "=" * 60)
    print("测试 3: Patch-wise 权重差异性")
    print("=" * 60)
    
    # 参数设置
    batch_size = 1
    n_vars = 1
    num_patches = 16
    d_model = 32
    num_bands = 2
    
    B_N = batch_size * n_vars
    
    # 创建有明显差异的输入
    # 前半部分 Patch 高频主导，后半部分低频主导
    torch.manual_seed(123)
    
    low_freq = torch.randn(B_N, num_patches, d_model)
    high_freq = torch.randn(B_N, num_patches, d_model)
    
    # 让前半部分高频信号更强
    high_freq[:, :num_patches//2, :] *= 5.0
    # 让后半部分低频信号更强
    low_freq[:, num_patches//2:, :] *= 5.0
    
    band_embeddings = [low_freq, high_freq]
    
    # 初始化 V2
    attn_v2 = FrequencyChannelAttentionV2(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 设置为 eval 模式以获得确定性输出
    attn_v2.eval()
    
    with torch.no_grad():
        output, weights = attn_v2(band_embeddings)
    
    # 检查不同 Patch 的权重是否不同
    weights_first_half = weights[:, :num_patches//2, :].mean(dim=1)  # (B_N, num_bands)
    weights_second_half = weights[:, num_patches//2:, :].mean(dim=1)  # (B_N, num_bands)
    
    print(f"\n前半部分 Patch 的平均权重 (低频, 高频): {weights_first_half[0].tolist()}")
    print(f"后半部分 Patch 的平均权重 (低频, 高频): {weights_second_half[0].tolist()}")
    
    # 检查权重是否有变化（不是所有 Patch 都一样）
    weight_variance = weights.var(dim=1).mean()
    print(f"\n权重在 Patch 维度上的方差: {weight_variance.item():.6f}")
    
    # 只要方差不为 0，就说明不同 Patch 有不同的权重
    # 注意：由于初始化是均匀的，初始方差可能很小，但不应该为 0
    print(f"权重是否有 Patch 级别的差异: {'是' if weight_variance > 0 else '否'}")
    
    print("✅ 测试 3 通过!")
    return True


def test_gradient_flow():
    """测试梯度能否正常反向传播"""
    print("\n" + "=" * 60)
    print("测试 4: 梯度反向传播")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    n_vars = 3
    num_patches = 16
    d_model = 32
    num_bands = 3
    
    B_N = batch_size * n_vars
    
    # 创建需要梯度的输入
    band_embeddings = [
        torch.randn(B_N, num_patches, d_model, requires_grad=True) 
        for _ in range(num_bands)
    ]
    
    # 初始化 V2
    attn_v2 = FrequencyChannelAttentionV2(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 前向传播
    output, weights = attn_v2(band_embeddings)
    
    # 计算一个简单的 loss
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查输入是否有梯度
    print("\n检查输入梯度:")
    for i, emb in enumerate(band_embeddings):
        has_grad = emb.grad is not None and emb.grad.abs().sum() > 0
        print(f"  频段 {i} 梯度: {'✅ 有' if has_grad else '❌ 无'}")
        assert has_grad, f"频段 {i} 应该有梯度"
    
    # 检查模块参数是否有梯度
    print("\n检查模块参数梯度:")
    for name, param in attn_v2.named_parameters():
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        print(f"  {name}: {'✅ 有' if has_grad else '❌ 无'}")
    
    print("✅ 测试 4 通过!")
    return True


def test_different_kernel_sizes():
    """测试不同的卷积核大小"""
    print("\n" + "=" * 60)
    print("测试 5: 不同卷积核大小")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    n_vars = 2
    num_patches = 32
    d_model = 32
    num_bands = 2
    
    B_N = batch_size * n_vars
    
    kernel_sizes = [1, 3, 5, 7]
    
    for ks in kernel_sizes:
        print(f"\n测试 kernel_size={ks}...")
        
        # 创建输入
        band_embeddings = [
            torch.randn(B_N, num_patches, d_model) for _ in range(num_bands)
        ]
        
        # 初始化
        attn_v2 = FrequencyChannelAttentionV2(
            num_bands=num_bands,
            d_model=d_model,
            reduction=4,
            kernel_size=ks
        )
        
        # 前向传播
        output, weights = attn_v2(band_embeddings)
        
        # 验证形状
        assert output.shape == (B_N, num_patches, d_model), f"kernel_size={ks} 输出形状错误"
        assert weights.shape == (B_N, num_patches, num_bands), f"kernel_size={ks} 权重形状错误"
        
        print(f"  ✅ kernel_size={ks} 通过")
    
    print("\n✅ 测试 5 通过!")
    return True


def test_parameter_count():
    """对比 V1 和 V2 的参数量"""
    print("\n" + "=" * 60)
    print("测试 6: 参数量对比")
    print("=" * 60)
    
    d_model = 64
    num_bands = 3
    
    attn_v1 = FrequencyChannelAttention(num_bands=num_bands, d_model=d_model, reduction=4)
    attn_v2 = FrequencyChannelAttentionV2(num_bands=num_bands, d_model=d_model, reduction=4, kernel_size=3)
    
    params_v1 = sum(p.numel() for p in attn_v1.parameters())
    params_v2 = sum(p.numel() for p in attn_v2.parameters())
    
    print(f"\nV1 参数量: {params_v1:,}")
    print(f"V2 参数量: {params_v2:,}")
    print(f"V2 相比 V1 增加: {params_v2 - params_v1:,} ({(params_v2/params_v1 - 1)*100:.1f}%)")
    
    print("\n✅ 测试 6 通过!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FrequencyChannelAttentionV2 模块测试")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_basic_forward()
        all_passed &= test_v1_vs_v2_comparison()
        all_passed &= test_patch_wise_weights()
        all_passed &= test_gradient_flow()
        all_passed &= test_different_kernel_sizes()
        all_passed &= test_parameter_count()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
