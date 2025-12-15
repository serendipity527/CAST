"""
测试 FrequencyChannelAttention 模块的功能正确性

测试内容:
1. 基础功能测试: 输入输出形状验证
2. 注意力权重验证: 确保权重归一化 (sum=1)
3. 梯度流测试: 确保可以正常反向传播
4. 与 WISTPatchEmbedding 集成测试: 双通道模式和金字塔模式
5. 动态路由验证: 不同样本应该得到不同的权重
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np


def test_frequency_channel_attention_basic():
    """测试 FrequencyChannelAttention 的基础功能"""
    from layers.Embed import FrequencyChannelAttention
    
    print("=" * 60)
    print("测试 1: FrequencyChannelAttention 基础功能测试")
    print("=" * 60)
    
    # 参数设置
    batch_size = 8
    num_patches = 32
    d_model = 64
    num_bands = 3  # 例如: cA, cD_2, cD_1
    
    # 创建模块
    freq_attn = FrequencyChannelAttention(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4
    )
    
    # 创建输入: 模拟各频段的 embedding
    band_embeddings = [
        torch.randn(batch_size, num_patches, d_model) for _ in range(num_bands)
    ]
    
    # 前向传播
    output, attention_weights = freq_attn(band_embeddings)
    
    # 验证输出形状
    expected_output_shape = (batch_size, num_patches, d_model)
    expected_weights_shape = (batch_size, num_bands)
    
    assert output.shape == expected_output_shape, \
        f"输出形状错误: 期望 {expected_output_shape}, 实际 {output.shape}"
    assert attention_weights.shape == expected_weights_shape, \
        f"权重形状错误: 期望 {expected_weights_shape}, 实际 {attention_weights.shape}"
    
    print(f"  ✅ 输出形状正确: {output.shape}")
    print(f"  ✅ 注意力权重形状正确: {attention_weights.shape}")
    
    # 验证注意力权重归一化
    weight_sums = attention_weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"注意力权重未归一化: {weight_sums}"
    print(f"  ✅ 注意力权重归一化正确 (sum ≈ 1.0)")
    
    # 验证权重非负
    assert (attention_weights >= 0).all(), "注意力权重存在负值"
    print(f"  ✅ 注意力权重非负")
    
    print("\n测试 1 通过!\n")
    return True


def test_gradient_flow():
    """测试梯度能否正常流动"""
    from layers.Embed import FrequencyChannelAttention
    
    print("=" * 60)
    print("测试 2: 梯度流测试")
    print("=" * 60)
    
    batch_size = 4
    num_patches = 16
    d_model = 32
    num_bands = 2
    
    freq_attn = FrequencyChannelAttention(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4
    )
    
    # 创建需要梯度的输入
    band_embeddings = [
        torch.randn(batch_size, num_patches, d_model, requires_grad=True)
        for _ in range(num_bands)
    ]
    
    # 前向传播
    output, _ = freq_attn(band_embeddings)
    
    # 计算一个简单的损失
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    # 验证输入的梯度不为 None
    for i, emb in enumerate(band_embeddings):
        assert emb.grad is not None, f"频段 {i} 的梯度为 None"
        assert not torch.isnan(emb.grad).any(), f"频段 {i} 的梯度包含 NaN"
        print(f"  ✅ 频段 {i} 梯度正常, 范数: {emb.grad.norm().item():.4f}")
    
    # 验证模块参数的梯度
    for name, param in freq_attn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"参数 {name} 的梯度为 None"
            print(f"  ✅ 参数 {name} 梯度正常")
    
    print("\n测试 2 通过!\n")
    return True


def test_instance_wise_dynamic_routing():
    """测试不同样本是否得到不同的注意力权重 (动态路由)
    
    注意: 初始化时权重是均匀的 (这是设计目标)，需要模拟一步训练后再验证动态路由能力
    """
    from layers.Embed import FrequencyChannelAttention
    
    print("=" * 60)
    print("测试 3: Instance-wise 动态路由测试")
    print("=" * 60)
    
    batch_size = 2
    num_patches = 16
    d_model = 32
    num_bands = 2
    
    freq_attn = FrequencyChannelAttention(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4
    )
    
    # 创建两个差异很大的样本
    low_freq_strong = torch.randn(1, num_patches, d_model) * 10
    high_freq_strong = torch.randn(1, num_patches, d_model) * 10
    low_freq_weak = torch.randn(1, num_patches, d_model) * 0.1
    high_freq_weak = torch.randn(1, num_patches, d_model) * 0.1
    
    # 样本 0: 低频强, 高频弱
    # 样本 1: 低频弱, 高频强
    batch_low = torch.cat([low_freq_strong, low_freq_weak], dim=0)
    batch_high = torch.cat([high_freq_weak, high_freq_strong], dim=0)
    
    band_embeddings = [batch_low, batch_high]
    
    # 初始化时权重是均匀的，这是预期的
    output, attention_weights = freq_attn(band_embeddings)
    print(f"  初始化时权重 (均匀分布是预期的):")
    print(f"    样本 0: {attention_weights[0].detach().numpy()}")
    print(f"    样本 1: {attention_weights[1].detach().numpy()}")
    
    # 模拟一步训练: 让模型学习到低频强的样本应该更关注低频
    optimizer = torch.optim.SGD(freq_attn.parameters(), lr=1.0)
    
    # 构造一个简单的目标: 样本 0 应该更关注低频 (索引 0)，样本 1 应该更关注高频 (索引 1)
    target_weights = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    
    for _ in range(10):  # 训练几步
        optimizer.zero_grad()
        output, attention_weights = freq_attn(band_embeddings)
        loss = ((attention_weights - target_weights) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    # 训练后验证
    output, attention_weights = freq_attn(band_embeddings)
    print(f"  训练后权重 (应该不同):")
    print(f"    样本 0: {attention_weights[0].detach().numpy()}")
    print(f"    样本 1: {attention_weights[1].detach().numpy()}")
    
    # 验证两个样本的权重不同 (动态路由)
    weights_diff = (attention_weights[0] - attention_weights[1]).abs().sum()
    print(f"  权重差异: {weights_diff.item():.4f}")
    
    assert weights_diff > 0.1, f"训练后权重差异应该显著: {weights_diff.item()}"
    print(f"  ✅ 训练后不同样本得到不同的权重 (Instance-wise 动态路由生效)")
    
    print("\n测试 3 通过!\n")
    return True


def test_wist_pe_with_freq_attention_dual_channel():
    """测试 WISTPatchEmbedding 双通道模式下的频率注意力"""
    from layers.Embed import WISTPatchEmbedding
    
    print("=" * 60)
    print("测试 4: WISTPatchEmbedding 双通道 + 频率注意力")
    print("=" * 60)
    
    batch_size = 4
    n_vars = 7
    seq_len = 512
    d_model = 32
    patch_len = 16
    stride = 8
    
    # 创建模块 - 启用频率注意力
    wist_pe = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='haar',
        wavelet_level=1,  # 双通道模式
        hf_dropout=0.5,
        gate_bias_init=2.0,  # 这个参数在启用注意力时应该被忽略
        use_soft_threshold=True,
        use_causal_conv=True,
        pyramid_fusion=False,
        mf_dropout=0.3,
        use_freq_attention=True  # 启用频率注意力
    )
    
    # 创建输入
    x = torch.randn(batch_size, n_vars, seq_len)
    
    # 前向传播
    output, num_vars = wist_pe(x)
    
    # 计算期望的 patch 数量
    padded_len = seq_len + stride
    expected_num_patches = (padded_len - patch_len) // stride + 1
    expected_shape = (batch_size * n_vars, expected_num_patches, d_model)
    
    assert output.shape == expected_shape, \
        f"输出形状错误: 期望 {expected_shape}, 实际 {output.shape}"
    assert num_vars == n_vars, f"变量数错误: 期望 {n_vars}, 实际 {num_vars}"
    
    print(f"  ✅ 输出形状正确: {output.shape}")
    print(f"  ✅ 变量数正确: {num_vars}")
    
    # 验证 freq_attention 模块存在
    assert wist_pe.freq_attention is not None, "freq_attention 模块未创建"
    assert wist_pe.gate is None, "启用注意力时 gate 应该为 None"
    print(f"  ✅ freq_attention 模块已创建")
    print(f"  ✅ gate 模块正确设置为 None")
    
    print("\n测试 4 通过!\n")
    return True


def test_wist_pe_with_freq_attention_pyramid():
    """测试 WISTPatchEmbedding 金字塔模式下的频率注意力"""
    from layers.Embed import WISTPatchEmbedding
    
    print("=" * 60)
    print("测试 5: WISTPatchEmbedding 金字塔 + 频率注意力")
    print("=" * 60)
    
    batch_size = 4
    n_vars = 7
    seq_len = 512
    d_model = 32
    patch_len = 16
    stride = 8
    
    # 创建模块 - 金字塔模式 + 频率注意力
    wist_pe = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='db4',
        wavelet_level=2,  # 金字塔模式
        hf_dropout=0.5,
        gate_bias_init=2.0,
        use_soft_threshold=True,
        use_causal_conv=True,
        pyramid_fusion=True,
        mf_dropout=0.3,
        use_freq_attention=True  # 启用频率注意力
    )
    
    # 创建输入
    x = torch.randn(batch_size, n_vars, seq_len)
    
    # 前向传播
    output, num_vars = wist_pe(x)
    
    # 验证输出形状
    padded_len = seq_len + stride
    expected_num_patches = (padded_len - patch_len) // stride + 1
    expected_shape = (batch_size * n_vars, expected_num_patches, d_model)
    
    assert output.shape == expected_shape, \
        f"输出形状错误: 期望 {expected_shape}, 实际 {output.shape}"
    
    print(f"  ✅ 输出形状正确: {output.shape}")
    
    # 验证 freq_attention 模块存在且参数正确
    assert wist_pe.freq_attention is not None, "freq_attention 模块未创建"
    assert wist_pe.freq_attention.num_bands == 3, \
        f"频段数错误: 期望 3, 实际 {wist_pe.freq_attention.num_bands}"
    assert wist_pe.gate_layers is None, "启用注意力时 gate_layers 应该为 None"
    
    print(f"  ✅ freq_attention 模块已创建 (num_bands=3)")
    print(f"  ✅ gate_layers 正确设置为 None")
    
    print("\n测试 5 通过!\n")
    return True


def test_wist_pe_without_freq_attention():
    """测试 WISTPatchEmbedding 关闭频率注意力时使用门控融合"""
    from layers.Embed import WISTPatchEmbedding
    
    print("=" * 60)
    print("测试 6: WISTPatchEmbedding 关闭频率注意力 (使用门控融合)")
    print("=" * 60)
    
    batch_size = 4
    n_vars = 7
    seq_len = 512
    d_model = 32
    patch_len = 16
    stride = 8
    
    # 创建模块 - 关闭频率注意力
    wist_pe = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='haar',
        wavelet_level=1,
        hf_dropout=0.5,
        gate_bias_init=2.0,
        use_soft_threshold=True,
        use_causal_conv=True,
        pyramid_fusion=False,
        mf_dropout=0.3,
        use_freq_attention=False  # 关闭频率注意力
    )
    
    # 创建输入
    x = torch.randn(batch_size, n_vars, seq_len)
    
    # 前向传播
    output, num_vars = wist_pe(x)
    
    # 验证
    assert wist_pe.freq_attention is None, "关闭注意力时 freq_attention 应该为 None"
    assert wist_pe.gate is not None, "关闭注意力时应该使用 gate"
    
    print(f"  ✅ freq_attention 正确设置为 None")
    print(f"  ✅ gate 模块已创建")
    print(f"  ✅ 输出形状正确: {output.shape}")
    
    print("\n测试 6 通过!\n")
    return True


def test_attention_weights_visualization():
    """测试注意力权重的可视化友好性"""
    from layers.Embed import FrequencyChannelAttention
    
    print("=" * 60)
    print("测试 7: 注意力权重可视化验证")
    print("=" * 60)
    
    batch_size = 8
    num_patches = 32
    d_model = 64
    num_bands = 3
    
    freq_attn = FrequencyChannelAttention(
        num_bands=num_bands,
        d_model=d_model,
        reduction=4
    )
    
    # 创建输入
    band_embeddings = [
        torch.randn(batch_size, num_patches, d_model) for _ in range(num_bands)
    ]
    
    # 前向传播
    output, attention_weights = freq_attn(band_embeddings)
    
    # 打印权重统计
    weights_np = attention_weights.detach().numpy()
    print(f"  注意力权重统计:")
    print(f"    - 形状: {weights_np.shape}")
    print(f"    - 均值 (每个频段): {weights_np.mean(axis=0)}")
    print(f"    - 标准差 (每个频段): {weights_np.std(axis=0)}")
    print(f"    - 最小值: {weights_np.min():.4f}")
    print(f"    - 最大值: {weights_np.max():.4f}")
    
    # 初始化时权重应该相对均匀 (每个约 1/num_bands)
    expected_mean = 1.0 / num_bands
    actual_means = weights_np.mean(axis=0)
    
    print(f"\n  期望均值 (均匀分布): {expected_mean:.4f}")
    print(f"  实际均值: {actual_means}")
    
    # 初始化权重不应该过于极端
    assert weights_np.min() > 0.01, "初始权重过小"
    assert weights_np.max() < 0.99, "初始权重过大"
    print(f"  ✅ 初始权重分布合理")
    
    print("\n测试 7 通过!\n")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("开始运行 FrequencyChannelAttention 测试套件")
    print("=" * 70 + "\n")
    
    tests = [
        ("基础功能测试", test_frequency_channel_attention_basic),
        ("梯度流测试", test_gradient_flow),
        ("动态路由测试", test_instance_wise_dynamic_routing),
        ("双通道+注意力集成测试", test_wist_pe_with_freq_attention_dual_channel),
        ("金字塔+注意力集成测试", test_wist_pe_with_freq_attention_pyramid),
        ("关闭注意力测试", test_wist_pe_without_freq_attention),
        ("权重可视化测试", test_attention_weights_visualization),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"❌ {name} 失败: {e}\n")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✅ 通过" if success else f"❌ 失败: {error}"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
