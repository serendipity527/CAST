"""
测试CWPR实现的正确性

验证内容：
1. WIST层的forward_separated方法能正确输出分离的特征
2. 高频融合使用频率注意力V1版本
3. CWPR模块的维度正确性
4. 整个流程的端到端测试
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.Embed import WISTPatchEmbedding
from layers.CWPR import CWPRReprogrammingLayer, PrototypeBank, CWPRCrossAttention, SemanticGate


def test_wist_forward_separated():
    """测试WIST层的forward_separated方法"""
    print("=" * 70)
    print("测试 1: WIST forward_separated 方法")
    print("=" * 70)
    
    # 创建WIST层（金字塔融合模式，level=2）
    d_model = 32
    patch_len = 16
    stride = 8
    B, N, T = 2, 7, 96
    
    wist = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='db4',
        wavelet_level=2,
        pyramid_fusion=True,
        use_freq_attention=False,  # 不使用全频段注意力，但高频融合会使用V1
        use_soft_threshold=True,
        use_causal_conv=True
    )
    
    # 创建输入
    x = torch.randn(B, N, T)
    
    # 测试forward_separated
    e_cA, e_detail, n_vars = wist.forward_separated(x)
    
    # 验证输出
    assert e_cA.shape[0] == B * N, f"e_cA batch维度错误: {e_cA.shape[0]} != {B * N}"
    assert e_detail.shape[0] == B * N, f"e_detail batch维度错误: {e_detail.shape[0]} != {B * N}"
    assert e_cA.shape[1] == e_detail.shape[1], f"Patch数量不一致: {e_cA.shape[1]} != {e_detail.shape[1]}"
    assert e_cA.shape[2] == d_model, f"e_cA特征维度错误: {e_cA.shape[2]} != {d_model}"
    assert e_detail.shape[2] == d_model, f"e_detail特征维度错误: {e_detail.shape[2]} != {d_model}"
    assert n_vars == N, f"n_vars错误: {n_vars} != {N}"
    
    # 验证高频融合使用了频率注意力
    assert hasattr(wist, 'hf_freq_attention'), "hf_freq_attention未创建"
    assert wist.hf_freq_attention is not None, "hf_freq_attention为None"
    assert isinstance(wist.hf_freq_attention, nn.Module), "hf_freq_attention不是nn.Module"
    
    print(f"✓ e_cA形状: {e_cA.shape}")
    print(f"✓ e_detail形状: {e_detail.shape}")
    print(f"✓ n_vars: {n_vars}")
    print(f"✓ 高频融合使用频率注意力V1: {type(wist.hf_freq_attention).__name__}")
    print("✓ 测试通过！\n")


def test_wist_dual_channel_separated():
    """测试双通道模式的forward_separated"""
    print("=" * 70)
    print("测试 2: WIST 双通道模式 forward_separated")
    print("=" * 70)
    
    d_model = 32
    patch_len = 16
    stride = 8
    B, N, T = 2, 7, 96
    
    wist = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='db4',
        wavelet_level=1,  # level=1，双通道模式
        pyramid_fusion=False,
        use_soft_threshold=True,
        use_causal_conv=True
    )
    
    x = torch.randn(B, N, T)
    e_cA, e_detail, n_vars = wist.forward_separated(x)
    
    assert e_cA.shape == e_detail.shape, "e_cA和e_detail形状应该相同"
    assert e_cA.shape[2] == d_model, "特征维度错误"
    assert wist.hf_freq_attention is None, "双通道模式下hf_freq_attention应该为None"
    
    print(f"✓ e_cA形状: {e_cA.shape}")
    print(f"✓ e_detail形状: {e_detail.shape}")
    print("✓ 测试通过！\n")


def test_prototype_bank():
    """测试原型库"""
    print("=" * 70)
    print("测试 3: PrototypeBank")
    print("=" * 70)
    
    K = 256
    d_llm = 4096
    
    # 测试随机初始化
    proto_bank = PrototypeBank(
        num_prototypes=K,
        d_llm=d_llm,
        init_method='random'
    )
    
    prototypes = proto_bank()
    assert prototypes.shape == (K, d_llm), f"原型库形状错误: {prototypes.shape} != {(K, d_llm)}"
    
    # 测试word_embed初始化
    word_embeddings = torch.randn(50000, d_llm)
    proto_bank2 = PrototypeBank(
        num_prototypes=K,
        d_llm=d_llm,
        init_method='word_embed',
        word_embeddings=word_embeddings
    )
    
    prototypes2 = proto_bank2()
    assert prototypes2.shape == (K, d_llm), "word_embed初始化形状错误"
    
    print(f"✓ 随机初始化原型库形状: {prototypes.shape}")
    print(f"✓ word_embed初始化原型库形状: {prototypes2.shape}")
    print("✓ 测试通过！\n")


def test_cwpr_cross_attention():
    """测试CWPR Cross-Attention"""
    print("=" * 70)
    print("测试 4: CWPRCrossAttention")
    print("=" * 70)
    
    d_model = 32
    d_llm = 4096
    n_heads = 8
    K = 256
    B_N, P = 14, 12  # B*N=14, P=12
    
    cross_attn = CWPRCrossAttention(
        d_model=d_model,
        d_llm=d_llm,
        n_heads=n_heads
    )
    
    # 创建输入
    query = torch.randn(B_N, P, d_model)
    prototypes = torch.randn(K, d_llm)
    
    # 前向传播
    output = cross_attn(query, prototypes)
    
    assert output.shape == (B_N, P, d_llm), f"输出形状错误: {output.shape} != {(B_N, P, d_llm)}"
    
    print(f"✓ 输入Query形状: {query.shape}")
    print(f"✓ 原型库形状: {prototypes.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print("✓ 测试通过！\n")


def test_semantic_gate():
    """测试语义门控网络"""
    print("=" * 70)
    print("测试 5: SemanticGate")
    print("=" * 70)
    
    d_model = 32
    B_N, P = 14, 12
    
    gate = SemanticGate(d_model=d_model, gate_bias_init=2.0)
    
    e_cA = torch.randn(B_N, P, d_model)
    e_detail = torch.randn(B_N, P, d_model)
    
    gate_weights = gate(e_cA, e_detail)
    
    assert gate_weights.shape == (B_N, P, 1), f"门控权重形状错误: {gate_weights.shape} != {(B_N, P, 1)}"
    assert (gate_weights >= 0).all() and (gate_weights <= 1).all(), "门控权重应该在[0,1]范围内"
    
    # 检查初始偏置效果（应该偏向趋势，即gate值较大）
    mean_gate = gate_weights.mean().item()
    print(f"✓ 平均门控权重: {mean_gate:.4f} (应该接近sigmoid(2.0)≈0.88)")
    
    print(f"✓ 输入e_cA形状: {e_cA.shape}")
    print(f"✓ 输入e_detail形状: {e_detail.shape}")
    print(f"✓ 输出门控权重形状: {gate_weights.shape}")
    print("✓ 测试通过！\n")


def test_cwpr_reprogramming_layer():
    """测试CWPR重编程层"""
    print("=" * 70)
    print("测试 6: CWPRReprogrammingLayer")
    print("=" * 70)
    
    d_model = 32
    d_llm = 4096
    n_heads = 8
    K = 256
    B_N, P = 14, 12
    
    cwpr = CWPRReprogrammingLayer(
        d_model=d_model,
        d_llm=d_llm,
        n_heads=n_heads,
        num_prototypes=K,
        attention_dropout=0.1,
        gate_bias_init=2.0
    )
    
    e_cA = torch.randn(B_N, P, d_model)
    e_detail = torch.randn(B_N, P, d_model)
    
    output = cwpr(e_cA, e_detail)
    
    assert output.shape == (B_N, P, d_llm), f"输出形状错误: {output.shape} != {(B_N, P, d_llm)}"
    
    print(f"✓ 输入e_cA形状: {e_cA.shape}")
    print(f"✓ 输入e_detail形状: {e_detail.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print("✓ 测试通过！\n")


def test_end_to_end():
    """端到端测试：从输入到CWPR输出"""
    print("=" * 70)
    print("测试 7: 端到端测试 (WIST + CWPR)")
    print("=" * 70)
    
    # 参数
    d_model = 32
    d_llm = 4096
    n_heads = 8
    K = 256
    patch_len = 16
    stride = 8
    B, N, T = 2, 7, 96
    
    # 创建WIST层
    wist = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='db4',
        wavelet_level=2,
        pyramid_fusion=True,
        use_soft_threshold=True,
        use_causal_conv=True
    )
    
    # 创建CWPR层
    cwpr = CWPRReprogrammingLayer(
        d_model=d_model,
        d_llm=d_llm,
        n_heads=n_heads,
        num_prototypes=K,
        attention_dropout=0.1,
        gate_bias_init=2.0
    )
    
    # 输入
    x = torch.randn(B, N, T)
    
    # WIST层处理
    e_cA, e_detail, n_vars = wist.forward_separated(x)
    
    # CWPR层处理
    semantic_out = cwpr(e_cA, e_detail)
    
    # 验证
    assert semantic_out.shape[0] == B * N, "Batch维度错误"
    assert semantic_out.shape[1] == e_cA.shape[1], "Patch数量不一致"
    assert semantic_out.shape[2] == d_llm, "LLM维度错误"
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ WIST输出 e_cA: {e_cA.shape}")
    print(f"✓ WIST输出 e_detail: {e_detail.shape}")
    print(f"✓ CWPR输出: {semantic_out.shape}")
    print("✓ 端到端测试通过！\n")


def test_frequency_attention_v1_usage():
    """测试高频融合确实使用了频率注意力V1"""
    print("=" * 70)
    print("测试 8: 验证高频融合使用频率注意力V1")
    print("=" * 70)
    
    d_model = 32
    patch_len = 16
    stride = 8
    B, N, T = 2, 7, 96
    
    # 创建WIST层（level=2，应该有2个高频频段）
    wist = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='db4',
        wavelet_level=2,
        pyramid_fusion=True,
        use_freq_attention=False,  # 不使用全频段注意力
        use_soft_threshold=True,
        use_causal_conv=True
    )
    
    # 验证hf_freq_attention存在且是FrequencyChannelAttention类型
    assert hasattr(wist, 'hf_freq_attention'), "hf_freq_attention属性不存在"
    assert wist.hf_freq_attention is not None, "hf_freq_attention为None"
    
    # 检查类型
    from layers.Embed import FrequencyChannelAttention
    assert isinstance(wist.hf_freq_attention, FrequencyChannelAttention), \
        f"hf_freq_attention类型错误: {type(wist.hf_freq_attention)}，应该是FrequencyChannelAttention"
    
    # 验证num_bands正确（高频频段数量 = level = 2）
    assert wist.hf_freq_attention.num_bands == 2, \
        f"高频频段数量错误: {wist.hf_freq_attention.num_bands} != 2"
    
    # 测试前向传播
    x = torch.randn(B, N, T)
    e_cA, e_detail, n_vars = wist.forward_separated(x)
    
    # 验证输出形状正确
    assert e_detail.shape == e_cA.shape, "e_detail和e_cA形状应该相同"
    
    print(f"✓ hf_freq_attention类型: {type(wist.hf_freq_attention).__name__}")
    print(f"✓ 高频频段数量: {wist.hf_freq_attention.num_bands}")
    print(f"✓ e_detail形状: {e_detail.shape}")
    print("✓ 高频融合使用频率注意力V1验证通过！\n")


def test_backward_compatibility():
    """测试向后兼容性：原有forward方法仍然可用"""
    print("=" * 70)
    print("测试 9: 向后兼容性测试")
    print("=" * 70)
    
    d_model = 32
    patch_len = 16
    stride = 8
    B, N, T = 2, 7, 96
    
    wist = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='db4',
        wavelet_level=2,
        pyramid_fusion=True,
        use_soft_threshold=True,
        use_causal_conv=True
    )
    
    x = torch.randn(B, N, T)
    
    # 测试原有forward方法
    e_fused, n_vars_old = wist.forward(x)
    
    # 测试新的forward_separated方法
    e_cA, e_detail, n_vars_new = wist.forward_separated(x)
    
    # 验证n_vars一致
    assert n_vars_old == n_vars_new == N, "n_vars不一致"
    
    # 验证形状
    assert e_fused.shape[0] == B * N, "Batch维度错误"
    assert e_fused.shape[2] == d_model, "特征维度错误"
    
    print(f"✓ 原有forward输出形状: {e_fused.shape}")
    print(f"✓ forward_separated输出形状: e_cA={e_cA.shape}, e_detail={e_detail.shape}")
    print(f"✓ n_vars一致: {n_vars_old} == {n_vars_new}")
    print("✓ 向后兼容性测试通过！\n")


def test_gradient_flow():
    """测试梯度流"""
    print("=" * 70)
    print("测试 10: 梯度流测试")
    print("=" * 70)
    
    d_model = 32
    d_llm = 4096
    n_heads = 8
    K = 256
    patch_len = 16
    stride = 8
    B, N, T = 2, 7, 96
    
    wist = WISTPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        wavelet_type='db4',
        wavelet_level=2,
        pyramid_fusion=True,
        use_soft_threshold=True,
        use_causal_conv=True
    )
    
    cwpr = CWPRReprogrammingLayer(
        d_model=d_model,
        d_llm=d_llm,
        n_heads=n_heads,
        num_prototypes=K,
        attention_dropout=0.1
    )
    
    x = torch.randn(B, N, T, requires_grad=True)
    
    # 前向传播
    e_cA, e_detail, _ = wist.forward_separated(x)
    output = cwpr(e_cA, e_detail)
    
    # 反向传播
    loss = output.mean()
    loss.backward()
    
    # 验证梯度存在
    assert x.grad is not None, "输入梯度不存在"
    assert e_cA.grad is None or e_cA.requires_grad == False, "e_cA不应该有梯度（中间变量）"
    
    # 验证参数有梯度
    has_grad = False
    for param in list(wist.parameters()) + list(cwpr.parameters()):
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "至少有一个参数应该有梯度"
    
    print(f"✓ 输入梯度形状: {x.grad.shape}")
    print(f"✓ 参数梯度存在: {has_grad}")
    print("✓ 梯度流测试通过！\n")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CWPR 实现正确性测试")
    print("=" * 70 + "\n")
    
    try:
        test_wist_forward_separated()
        test_wist_dual_channel_separated()
        test_prototype_bank()
        test_cwpr_cross_attention()
        test_semantic_gate()
        test_cwpr_reprogramming_layer()
        test_end_to_end()
        test_frequency_attention_v1_usage()
        test_backward_compatibility()
        test_gradient_flow()
        
        print("=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

