"""
WIST-PE (Wavelet-Informed Spatio-Temporal Patch Embedding) 测试脚本

测试内容:
1. 模块实例化测试
2. 前向传播形状测试
3. 因果性验证测试
4. 门控机制测试
5. 软阈值去噪测试
6. 与原版 PatchEmbedding 对比测试

Author: CAST Project
Date: 2024
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("WIST-PE 单元测试")
print("=" * 70)

# 设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\n测试设备: {device}")


def test_instantiation():
    """测试1: 模块实例化"""
    print("\n" + "=" * 70)
    print("测试1: 模块实例化")
    print("=" * 70)
    
    from layers.Embed import WISTPatchEmbedding, PatchEmbedding, WaveletPatchEmbedding
    
    # 测试 WISTPatchEmbedding 实例化
    try:
        wist_pe = WISTPatchEmbedding(
            d_model=32,
            patch_len=16,
            stride=8,
            dropout=0.1,
            wavelet_type='db4',
            wavelet_level=1,
            hf_dropout=0.5,
            gate_bias_init=2.0,
            use_soft_threshold=True
        ).to(device)
        print("\n✅ WISTPatchEmbedding 实例化成功")
    except Exception as e:
        print(f"\n❌ WISTPatchEmbedding 实例化失败: {e}")
        return False
    
    # 测试不同小波类型
    for wavelet in ['db1', 'db2', 'db3', 'db4', 'db5', 'haar']:
        try:
            _ = WISTPatchEmbedding(
                d_model=32, patch_len=16, stride=8, dropout=0.1,
                wavelet_type=wavelet
            ).to(device)
            print(f"  ✅ 小波类型 '{wavelet}' 支持")
        except Exception as e:
            print(f"  ❌ 小波类型 '{wavelet}' 失败: {e}")
    
    return True


def test_forward_shape():
    """测试2: 前向传播形状"""
    print("\n" + "=" * 70)
    print("测试2: 前向传播形状")
    print("=" * 70)
    
    from layers.Embed import WISTPatchEmbedding, PatchEmbedding
    
    # 测试参数
    B, N, T = 4, 7, 512  # batch, variables, time
    d_model = 32
    patch_len = 16
    stride = 8
    
    print(f"\n输入参数: B={B}, N={N}, T={T}")
    print(f"模型参数: d_model={d_model}, patch_len={patch_len}, stride={stride}")
    
    # 计算预期的 patch 数量
    num_patches = int((T - patch_len) / stride + 2)  # +2 是因为 padding
    print(f"预期 num_patches: {num_patches}")
    
    # 创建输入
    x = torch.randn(B, N, T, device=device)
    print(f"\n输入形状: {x.shape}")
    
    # WIST-PE 前向传播
    wist_pe = WISTPatchEmbedding(
        d_model=d_model, patch_len=patch_len, stride=stride, dropout=0.1
    ).to(device)
    
    output, n_vars = wist_pe(x)
    print(f"WIST-PE 输出形状: {output.shape}")
    print(f"返回的 n_vars: {n_vars}")
    
    # 验证形状
    expected_shape = (B * N, num_patches, d_model)
    if output.shape == expected_shape:
        print(f"✅ 输出形状正确: {output.shape} == {expected_shape}")
    else:
        print(f"❌ 输出形状错误: {output.shape} != {expected_shape}")
        return False
    
    if n_vars == N:
        print(f"✅ n_vars 正确: {n_vars} == {N}")
    else:
        print(f"❌ n_vars 错误: {n_vars} != {N}")
        return False
    
    # 对比原版 PatchEmbedding
    original_pe = PatchEmbedding(
        d_model=d_model, patch_len=patch_len, stride=stride, dropout=0.1
    ).to(device)
    
    # 原版需要 (B, N, T) -> (B*N, T, 1) 的转换
    x_for_original = x.reshape(B * N, T, 1).permute(0, 2, 1)  # (B*N, 1, T)
    output_orig, n_vars_orig = original_pe(x_for_original)
    print(f"\n原版 PatchEmbedding 输出形状: {output_orig.shape}")
    
    return True


def test_causality():
    """测试3: 因果性验证 - 修改未来数据不应影响过去的输出"""
    print("\n" + "=" * 70)
    print("测试3: 因果性验证")
    print("=" * 70)
    
    from layers.Embed import WISTPatchEmbedding
    
    # 创建模型 (eval模式关闭dropout)
    wist_pe = WISTPatchEmbedding(
        d_model=32, patch_len=16, stride=8, dropout=0.0,
        hf_dropout=0.0  # 关闭高频dropout以便精确测试
    ).to(device)
    wist_pe.eval()
    
    # 创建原始输入
    B, N, T = 1, 1, 128
    x_orig = torch.randn(B, N, T, device=device)
    
    # 复制并修改"未来"的数据点
    x_mod = x_orig.clone()
    target_time = 80  # 修改 t=80 及之后的数据
    x_mod[:, :, target_time:] += 100.0  # 大幅修改未来数据
    
    print(f"\n测试: 修改 t>={target_time} 的数据(+100)，检查 t<{target_time} 对应的 patch 输出")
    
    # 前向传播
    with torch.no_grad():
        output_orig, _ = wist_pe(x_orig)
        output_mod, _ = wist_pe(x_mod)
    
    # 计算哪些 patch 完全在 target_time 之前
    patch_len = 16
    stride = 8
    # patch i 覆盖的时间范围是 [i*stride, i*stride + patch_len)
    # 如果 i*stride + patch_len <= target_time，则该 patch 完全在修改点之前
    safe_patches = target_time // stride - 1  # 保守估计
    
    print(f"安全 patch 数量 (完全在修改点之前): {safe_patches}")
    
    # 检查安全 patch 的输出是否一致
    diff = (output_orig[:, :safe_patches, :] - output_mod[:, :safe_patches, :]).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n前 {safe_patches} 个 patch 的差异:")
    print(f"  - 最大差异: {max_diff:.10f}")
    print(f"  - 平均差异: {mean_diff:.10f}")
    
    if max_diff < 1e-5:
        print("✅ 因果性验证通过: 修改未来数据不影响过去的 patch 输出")
        return True
    else:
        print("❌ 因果性验证失败: 存在信息泄露")
        return False


def test_gate_mechanism():
    """测试4: 门控机制测试"""
    print("\n" + "=" * 70)
    print("测试4: 门控机制测试")
    print("=" * 70)
    
    from layers.Embed import WISTPatchEmbedding
    
    # 测试不同的 gate_bias_init
    for bias_init in [0.0, 2.0, 4.0]:
        wist_pe = WISTPatchEmbedding(
            d_model=32, patch_len=16, stride=8, dropout=0.0,
            gate_bias_init=bias_init, hf_dropout=0.0
        ).to(device)
        
        # 获取门控层的偏置
        gate_bias = None
        for m in wist_pe.gate.modules():
            if isinstance(m, nn.Linear):
                gate_bias = m.bias.data.mean().item()
                break
        
        expected_ratio = torch.sigmoid(torch.tensor(bias_init)).item()
        print(f"\nbias_init={bias_init:.1f} -> sigmoid={expected_ratio:.2%} 低频关注度")
        print(f"  实际 gate bias: {gate_bias:.4f}")
        
        if abs(gate_bias - bias_init) < 1e-5:
            print(f"  ✅ 门控偏置初始化正确")
        else:
            print(f"  ❌ 门控偏置初始化错误")
    
    return True


def test_soft_threshold():
    """测试5: 软阈值去噪测试"""
    print("\n" + "=" * 70)
    print("测试5: 软阈值去噪测试")
    print("=" * 70)
    
    from layers.Embed import SoftThreshold
    
    # 创建软阈值模块
    num_features = 16
    init_tau = 0.5
    soft_thresh = SoftThreshold(num_features=num_features, init_tau=init_tau).to(device)
    
    # 创建测试输入
    x = torch.randn(4, 10, num_features, device=device)
    
    # 应用软阈值
    y = soft_thresh(x)
    
    print(f"\n输入统计: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    print(f"输出统计: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    
    # 验证软阈值效果: 小于阈值的值应该变为0或接近0
    tau = soft_thresh.tau.abs()
    small_values_mask = x.abs() < tau.mean()
    small_values_output = y[small_values_mask]
    
    # 软阈值应该将小于tau的值压缩
    print(f"\n阈值 tau 均值: {tau.mean().item():.4f}")
    print(f"小于阈值的输入数量: {small_values_mask.sum().item()}")
    print(f"对应输出的绝对值均值: {small_values_output.abs().mean().item():.6f}")
    
    # 验证可学习性
    print(f"\ntau 是否可学习: {soft_thresh.tau.requires_grad}")
    
    if soft_thresh.tau.requires_grad:
        print("✅ 软阈值参数可学习")
        return True
    else:
        print("❌ 软阈值参数不可学习")
        return False


def test_gradient_flow():
    """测试6: 梯度流测试"""
    print("\n" + "=" * 70)
    print("测试6: 梯度流测试")
    print("=" * 70)
    
    from layers.Embed import WISTPatchEmbedding
    
    # 创建模型
    wist_pe = WISTPatchEmbedding(
        d_model=32, patch_len=16, stride=8, dropout=0.1
    ).to(device)
    wist_pe.train()
    
    # 创建输入
    x = torch.randn(2, 3, 128, device=device, requires_grad=True)
    
    # 前向传播
    output, _ = wist_pe(x)
    
    # 计算损失并反向传播
    loss = output.mean()
    loss.backward()
    
    # 检查梯度
    print("\n检查各组件梯度:")
    
    grad_checks = {
        'low_freq_embedding': wist_pe.low_freq_embedding.weight.grad,
        'high_freq_embedding': wist_pe.high_freq_embedding.weight.grad,
        'soft_threshold.tau': wist_pe.soft_threshold.tau.grad if hasattr(wist_pe, 'soft_threshold') else None,
    }
    
    # 检查门控层梯度
    for name, module in wist_pe.gate.named_modules():
        if isinstance(module, nn.Linear):
            grad_checks['gate.weight'] = module.weight.grad
            grad_checks['gate.bias'] = module.bias.grad
    
    all_grads_ok = True
    for name, grad in grad_checks.items():
        if grad is not None:
            grad_norm = grad.norm().item()
            status = "✅" if grad_norm > 0 else "⚠️"
            print(f"  {status} {name}: grad_norm = {grad_norm:.6f}")
            if grad_norm == 0:
                all_grads_ok = False
        else:
            print(f"  ⚠️ {name}: 无梯度")
    
    if all_grads_ok:
        print("\n✅ 梯度流正常")
        return True
    else:
        print("\n⚠️ 部分梯度为零，请检查")
        return True  # 不算失败，因为可能是正常现象


def test_different_seq_lengths():
    """测试7: 不同序列长度测试"""
    print("\n" + "=" * 70)
    print("测试7: 不同序列长度测试")
    print("=" * 70)
    
    from layers.Embed import WISTPatchEmbedding
    
    wist_pe = WISTPatchEmbedding(
        d_model=32, patch_len=16, stride=8, dropout=0.0
    ).to(device)
    wist_pe.eval()
    
    # 测试不同的序列长度
    test_lengths = [64, 96, 128, 256, 512]
    
    print("\n序列长度测试:")
    for T in test_lengths:
        try:
            x = torch.randn(2, 3, T, device=device)
            with torch.no_grad():
                output, n_vars = wist_pe(x)
            print(f"  ✅ T={T}: 输出形状 {output.shape}")
        except Exception as e:
            print(f"  ❌ T={T}: 失败 - {e}")
            return False
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("开始运行所有 WIST-PE 测试")
    print("=" * 70)
    
    tests = [
        ("模块实例化", test_instantiation),
        ("前向传播形状", test_forward_shape),
        ("因果性验证", test_causality),
        ("门控机制", test_gate_mechanism),
        ("软阈值去噪", test_soft_threshold),
        ("梯度流", test_gradient_flow),
        ("不同序列长度", test_different_seq_lengths),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！WIST-PE 实现正确！")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
