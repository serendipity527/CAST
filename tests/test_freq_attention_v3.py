"""
FrequencyChannelAttentionV3 (Global-Local 双流融合) 测试文件

测试内容:
1. 基础前向传播测试
2. V1 vs V2 vs V3 对比测试
3. 可学习 alpha 参数测试
4. 梯度流测试
5. Global/Local 分支独立性测试
6. 参数量对比测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np


def test_v3_basic_forward():
    """测试 V3 基础前向传播"""
    print("\n" + "=" * 70)
    print("测试 1: V3 基础前向传播")
    print("=" * 70)
    
    from layers.Embed import FrequencyChannelAttentionV3
    
    # 参数
    batch_size = 4
    num_patches = 32
    d_model = 64
    num_bands = 3
    
    # 创建模块
    v3 = FrequencyChannelAttentionV3(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 创建输入
    band_embeddings = [
        torch.randn(batch_size, num_patches, d_model)
        for _ in range(num_bands)
    ]
    
    # 前向传播
    output, attention_weights, fusion_info = v3(band_embeddings)
    
    # 检查输出形状
    assert output.shape == (batch_size, num_patches, d_model), \
        f"输出形状错误: 期望 {(batch_size, num_patches, d_model)}, 实际 {output.shape}"
    assert attention_weights.shape == (batch_size, num_patches, num_bands), \
        f"权重形状错误: 期望 {(batch_size, num_patches, num_bands)}, 实际 {attention_weights.shape}"
    
    # 检查权重和为 1
    weight_sum = attention_weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), \
        "权重和不为 1"
    
    # 检查 fusion_info
    assert 'alpha' in fusion_info, "fusion_info 缺少 alpha"
    assert 'global_weights' in fusion_info, "fusion_info 缺少 global_weights"
    assert 'local_weights' in fusion_info, "fusion_info 缺少 local_weights"
    
    print(f"✅ 输出形状正确: {output.shape}")
    print(f"✅ 权重形状正确: {attention_weights.shape}")
    print(f"✅ 权重和为 1")
    print(f"✅ Alpha 值: {fusion_info['alpha']:.4f}")
    print("✅ 测试通过!")
    
    return True


def test_v1_v2_v3_comparison():
    """对比 V1, V2, V3 的输出"""
    print("\n" + "=" * 70)
    print("测试 2: V1 vs V2 vs V3 对比")
    print("=" * 70)
    
    from layers.Embed import (
        FrequencyChannelAttention,
        FrequencyChannelAttentionV2,
        FrequencyChannelAttentionV3
    )
    
    # 参数
    batch_size = 4
    num_patches = 32
    d_model = 64
    num_bands = 3
    
    # 创建三个版本
    v1 = FrequencyChannelAttention(num_bands=num_bands, d_model=d_model, reduction=4)
    v2 = FrequencyChannelAttentionV2(num_bands=num_bands, d_model=d_model, reduction=4, kernel_size=3)
    v3 = FrequencyChannelAttentionV3(num_bands=num_bands, d_model=d_model, reduction=4, kernel_size=3)
    
    # 用随机值初始化 V2/V3 的 MLP 最后一层，以便测试 Patch-wise 特性
    # (默认初始化为 0 是为了让初始权重均匀，但这里需要验证架构能力)
    with torch.no_grad():
        nn.init.normal_(v2.excitation[-1].weight, std=0.1)
        nn.init.normal_(v2.excitation[-1].bias, std=0.1)
        nn.init.normal_(v3.local_excitation[-1].weight, std=0.1)
        nn.init.normal_(v3.local_excitation[-1].bias, std=0.1)
    
    # 相同输入
    torch.manual_seed(42)
    band_embeddings = [
        torch.randn(batch_size, num_patches, d_model)
        for _ in range(num_bands)
    ]
    
    # 前向传播
    out_v1, weights_v1 = v1(band_embeddings)
    out_v2, weights_v2 = v2(band_embeddings)
    out_v3, weights_v3, fusion_info = v3(band_embeddings)
    
    # 检查输出形状
    assert out_v1.shape == out_v2.shape == out_v3.shape, "输出形状不一致"
    
    # V1 权重是 Instance-wise (所有 Patch 共享)
    # V2/V3 权重是 Patch-wise (每个 Patch 独立)
    v1_weight_std = weights_v1.std(dim=1).mean().item()  # 沿 Patch 维度的标准差 (应该为 0)
    v2_weight_std = weights_v2.std(dim=1).mean().item()  # 应该 > 0
    v3_weight_std = weights_v3.std(dim=1).mean().item()  # 应该 > 0
    
    print(f"V1 权重 Patch 间标准差: {v1_weight_std:.6f} (应接近 0)")
    print(f"V2 权重 Patch 间标准差: {v2_weight_std:.6f} (应 > 0)")
    print(f"V3 权重 Patch 间标准差: {v3_weight_std:.6f} (应 > 0)")
    
    assert v1_weight_std < 1e-5, "V1 权重应该是 Instance-wise 的"
    assert v2_weight_std > 1e-5, "V2 权重应该是 Patch-wise 的"
    assert v3_weight_std > 1e-5, "V3 权重应该是 Patch-wise 的"
    
    print(f"✅ V1 是 Instance-wise (权重跨 Patch 一致)")
    print(f"✅ V2 是 Patch-wise (权重跨 Patch 变化)")
    print(f"✅ V3 是 Patch-wise (权重跨 Patch 变化)")
    print("✅ 测试通过!")
    
    return True


def test_learnable_alpha():
    """测试可学习的 alpha 参数"""
    print("\n" + "=" * 70)
    print("测试 3: 可学习 Alpha 参数")
    print("=" * 70)
    
    from layers.Embed import FrequencyChannelAttentionV3
    
    # 参数
    batch_size = 4
    num_patches = 32
    d_model = 64
    num_bands = 3
    
    # 创建模块
    v3 = FrequencyChannelAttentionV3(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 检查 alpha 是否是可学习参数
    alpha_found = False
    for name, param in v3.named_parameters():
        if 'alpha' in name:
            alpha_found = True
            print(f"✅ 找到 alpha 参数: {name}, 值={param.item():.4f}, requires_grad={param.requires_grad}")
            assert param.requires_grad, "Alpha 应该是可学习的"
    
    assert alpha_found, "未找到 alpha 参数"
    
    # 模拟训练: 检查 alpha 是否会更新
    optimizer = torch.optim.SGD(v3.parameters(), lr=0.1)
    
    band_embeddings = [
        torch.randn(batch_size, num_patches, d_model)
        for _ in range(num_bands)
    ]
    target = torch.randn(batch_size, num_patches, d_model)
    
    initial_alpha = v3.alpha.item()
    
    # 多次梯度更新
    for _ in range(10):
        optimizer.zero_grad()
        output, _, _ = v3(band_embeddings)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    updated_alpha = v3.alpha.item()
    
    print(f"初始 alpha: {initial_alpha:.4f}")
    print(f"更新后 alpha: {updated_alpha:.4f}")
    print(f"变化量: {abs(updated_alpha - initial_alpha):.6f}")
    
    assert abs(updated_alpha - initial_alpha) > 1e-6, "Alpha 应该在训练中更新"
    print("✅ Alpha 在训练中成功更新!")
    print("✅ 测试通过!")
    
    return True


def test_gradient_flow():
    """测试梯度流通"""
    print("\n" + "=" * 70)
    print("测试 4: 梯度流通")
    print("=" * 70)
    
    from layers.Embed import FrequencyChannelAttentionV3
    
    # 参数
    batch_size = 4
    num_patches = 32
    d_model = 64
    num_bands = 3
    
    # 创建模块
    v3 = FrequencyChannelAttentionV3(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 创建需要梯度的输入
    band_embeddings = [
        torch.randn(batch_size, num_patches, d_model, requires_grad=True)
        for _ in range(num_bands)
    ]
    
    # 前向传播
    output, _, _ = v3(band_embeddings)
    
    # 反向传播
    loss = output.sum()
    loss.backward()
    
    # 检查梯度
    for i, emb in enumerate(band_embeddings):
        assert emb.grad is not None, f"频段 {i} 没有梯度"
        assert not torch.isnan(emb.grad).any(), f"频段 {i} 梯度包含 NaN"
        grad_norm = emb.grad.norm().item()
        print(f"频段 {i} 梯度范数: {grad_norm:.6f}")
    
    # 检查模型参数梯度
    for name, param in v3.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"参数 {name}: 梯度范数={grad_norm:.6f}")
            assert not torch.isnan(param.grad).any(), f"参数 {name} 梯度包含 NaN"
    
    print("✅ 所有梯度正常流通!")
    print("✅ 测试通过!")
    
    return True


def test_global_local_decomposition():
    """测试 Global 和 Local 分支的分解效果"""
    print("\n" + "=" * 70)
    print("测试 5: Global/Local 分支分解效果")
    print("=" * 70)
    
    from layers.Embed import FrequencyChannelAttentionV3
    
    # 参数
    batch_size = 4
    num_patches = 32
    d_model = 64
    num_bands = 3
    
    # 创建模块
    v3 = FrequencyChannelAttentionV3(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 用随机值初始化 Local MLP 最后一层，以便测试 Patch-wise 特性
    # (默认初始化为 0 是为了让初始权重均匀，但这里需要验证架构能力)
    with torch.no_grad():
        nn.init.normal_(v3.local_excitation[-1].weight, std=0.1)
        nn.init.normal_(v3.local_excitation[-1].bias, std=0.1)
    
    # 创建输入
    torch.manual_seed(42)
    band_embeddings = [
        torch.randn(batch_size, num_patches, d_model)
        for _ in range(num_bands)
    ]
    
    # 前向传播
    output, attention_weights, fusion_info = v3(band_embeddings)
    
    # 检查 Global 权重 (应该在 Patch 维度上一致)
    global_weights = fusion_info['global_weights']
    global_std = global_weights.std(dim=1).mean().item()
    print(f"Global 权重 Patch 间标准差: {global_std:.6f} (应接近 0)")
    assert global_std < 1e-5, "Global 权重应该跨 Patch 一致"
    
    # 检查 Local 权重 (应该在 Patch 维度上变化)
    local_weights = fusion_info['local_weights']
    local_std = local_weights.std(dim=1).mean().item()
    print(f"Local 权重 Patch 间标准差: {local_std:.6f} (应 > 0)")
    assert local_std > 1e-5, "Local 权重应该跨 Patch 变化"
    
    # 检查融合权重 (介于 Global 和 Local 之间)
    fused_std = attention_weights.std(dim=1).mean().item()
    print(f"融合权重 Patch 间标准差: {fused_std:.6f}")
    
    print(f"✅ Global 分支: Instance-wise (跨 Patch 一致)")
    print(f"✅ Local 分支: Patch-wise (跨 Patch 变化)")
    print(f"✅ 融合权重: 结合了两者特性")
    print("✅ 测试通过!")
    
    return True


def test_parameter_count():
    """对比 V1, V2, V3 的参数量"""
    print("\n" + "=" * 70)
    print("测试 6: 参数量对比")
    print("=" * 70)
    
    from layers.Embed import (
        FrequencyChannelAttention,
        FrequencyChannelAttentionV2,
        FrequencyChannelAttentionV3
    )
    
    # 参数
    d_model = 64
    num_bands = 3
    
    v1 = FrequencyChannelAttention(num_bands=num_bands, d_model=d_model, reduction=4)
    v2 = FrequencyChannelAttentionV2(num_bands=num_bands, d_model=d_model, reduction=4, kernel_size=3)
    v3 = FrequencyChannelAttentionV3(num_bands=num_bands, d_model=d_model, reduction=4, kernel_size=3)
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    v1_params = count_params(v1)
    v2_params = count_params(v2)
    v3_params = count_params(v3)
    
    print(f"V1 (GAP) 参数量: {v1_params:,}")
    print(f"V2 (1D Conv) 参数量: {v2_params:,}")
    print(f"V3 (Global-Local) 参数量: {v3_params:,}")
    print(f"V3/V1 比例: {v3_params/v1_params:.2f}x")
    print(f"V3/V2 比例: {v3_params/v2_params:.2f}x")
    
    # V3 应该比 V2 多一些参数 (Global MLP)
    assert v3_params > v2_params, "V3 参数量应大于 V2"
    
    print("✅ 参数量符合预期!")
    print("✅ 测试通过!")
    
    return True


def test_alpha_range():
    """测试 alpha 值的范围限制"""
    print("\n" + "=" * 70)
    print("测试 7: Alpha 值范围限制")
    print("=" * 70)
    
    from layers.Embed import FrequencyChannelAttentionV3
    
    # 参数
    batch_size = 4
    num_patches = 32
    d_model = 64
    num_bands = 3
    
    # 创建模块
    v3 = FrequencyChannelAttentionV3(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4,
        kernel_size=3
    )
    
    # 测试极端 alpha 值
    test_values = [-10.0, -1.0, 0.0, 0.5, 1.0, 10.0]
    
    for val in test_values:
        v3.alpha.data = torch.tensor(val)
        
        band_embeddings = [
            torch.randn(batch_size, num_patches, d_model)
            for _ in range(num_bands)
        ]
        
        output, _, fusion_info = v3(band_embeddings)
        alpha = fusion_info['alpha']
        
        print(f"设置 alpha={val:.1f}, 实际使用 alpha={alpha:.4f}")
        
        # alpha 经过 sigmoid 后应该在 [0, 1] 范围内
        assert 0 <= alpha <= 1, f"Alpha 应该在 [0, 1] 范围内, 实际值: {alpha}"
    
    print("✅ Alpha 值始终在 [0, 1] 范围内!")
    print("✅ 测试通过!")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("FrequencyChannelAttentionV3 (Global-Local 双流融合) 完整测试")
    print("=" * 70)
    
    tests = [
        ("基础前向传播", test_v3_basic_forward),
        ("V1 vs V2 vs V3 对比", test_v1_v2_v3_comparison),
        ("可学习 Alpha 参数", test_learnable_alpha),
        ("梯度流通", test_gradient_flow),
        ("Global/Local 分支分解", test_global_local_decomposition),
        ("参数量对比", test_parameter_count),
        ("Alpha 值范围限制", test_alpha_range),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ 测试 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! FrequencyChannelAttentionV3 实现正确!")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查实现!")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
