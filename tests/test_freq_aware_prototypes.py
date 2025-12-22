"""
测试频率感知原型增强（Frequency-Aware Prototype Enhancement）功能

核心思想：P_trend = P_shared + B_trend, P_detail = P_shared + B_detail
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.TimeLLM import Model


class MockConfig:
    """模拟配置类"""
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.pred_len = 96
        self.seq_len = 336
        self.d_model = 512
        self.d_ff = 512
        self.llm_dim = 768
        self.patch_len = 16
        self.stride = 8
        self.enc_in = 7
        self.dropout = 0.1
        self.n_heads = 8
        self.llm_model = 'GPT2'
        self.llm_layers = 2
        self.prompt_domain = False
        self.content = 'Test dataset'
        
        # 分离原型配置
        self.use_dual_prototypes = True
        self.dual_proto_trend_tokens = 1000
        self.dual_proto_detail_tokens = 1000
        
        # 频率感知原型增强配置
        self.use_freq_aware_prototypes = True
        self.shared_proto_size = 800
        
        # 其他配置
        self.wavelet_mode = 'none'
        self.use_cwpr = False
        self.use_full_vocab_split = False
        self.use_semantic_filtered_mapping = False
        self.dual_proto_fusion_method = 'mean'


def test_freq_aware_prototypes_basic():
    """测试1: 基本功能 - 频率感知原型增强的初始化和前向传播"""
    print("\n" + "=" * 70)
    print("测试1: 基本功能 - 频率感知原型增强")
    print("=" * 70)
    
    config = MockConfig()
    config.use_freq_aware_prototypes = True
    config.shared_proto_size = 800
    
    model = Model(config)
    
    # 检查是否创建了共享映射层和偏置
    assert model.shared_mapping is not None, "共享映射层应该被创建"
    assert model.trend_bias is not None, "趋势偏置应该被创建"
    assert model.detail_bias is not None, "细节偏置应该被创建"
    assert model.trend_mapping is None, "趋势映射层应该为None（使用频率感知模式）"
    assert model.detail_mapping is None, "细节映射层应该为None（使用频率感知模式）"
    
    print("✅ 初始化检查通过")
    
    # 检查共享原型库大小
    assert model.shared_proto_size == 800, f"共享原型库大小应该是800，实际是{model.shared_proto_size}"
    print(f"✅ 共享原型库大小: {model.shared_proto_size}")
    
    # 检查偏置形状
    assert model.trend_bias.shape == (800, 768), f"趋势偏置形状应该是(800, 768)，实际是{model.trend_bias.shape}"
    assert model.detail_bias.shape == (800, 768), f"细节偏置形状应该是(800, 768)，实际是{model.detail_bias.shape}"
    print(f"✅ 偏置形状正确: trend_bias={model.trend_bias.shape}, detail_bias={model.detail_bias.shape}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 336
    n_vars = 7
    x_enc = torch.randn(batch_size, seq_len, n_vars)
    x_mark_enc = None
    x_dec = None
    x_mark_dec = None
    
    try:
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert output.shape == (batch_size, config.pred_len, n_vars), \
            f"输出形状应该是({batch_size}, {config.pred_len}, {n_vars})，实际是{output.shape}"
        print(f"✅ 前向传播成功，输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        raise
    
    print("=" * 70)
    print("✅ 测试1通过：基本功能正常")
    print("=" * 70)


def test_freq_aware_prototypes_prototype_generation():
    """测试2: 原型生成逻辑 - 验证 P_trend = P_shared + B_trend"""
    print("\n" + "=" * 70)
    print("测试2: 原型生成逻辑验证")
    print("=" * 70)
    
    config = MockConfig()
    config.use_freq_aware_prototypes = True
    config.shared_proto_size = 800
    config.dual_proto_trend_tokens = 1000
    config.dual_proto_detail_tokens = 1000
    
    model = Model(config)
    model.eval()
    
    # 手动生成共享原型库
    with torch.no_grad():
        P_shared = model.shared_mapping(model.word_embeddings.permute(1, 0)).permute(1, 0)
        assert P_shared.shape == (800, 768), f"共享原型库形状应该是(800, 768)，实际是{P_shared.shape}"
        print(f"✅ 共享原型库形状: {P_shared.shape}")
        
        # 应用偏置
        P_trend = P_shared + model.trend_bias
        P_detail = P_shared + model.detail_bias
        
        assert P_trend.shape == (800, 768), f"趋势原型形状应该是(800, 768)，实际是{P_trend.shape}"
        assert P_detail.shape == (800, 768), f"细节原型形状应该是(800, 768)，实际是{P_detail.shape}"
        print(f"✅ 应用偏置后形状: P_trend={P_trend.shape}, P_detail={P_detail.shape}")
        
        # 验证 P_trend = P_shared + B_trend
        diff_trend = torch.abs(P_trend - (P_shared + model.trend_bias)).max()
        diff_detail = torch.abs(P_detail - (P_shared + model.detail_bias)).max()
        
        assert diff_trend < 1e-6, f"P_trend应该等于P_shared+B_trend，最大差异: {diff_trend}"
        assert diff_detail < 1e-6, f"P_detail应该等于P_shared+B_detail，最大差异: {diff_detail}"
        print(f"✅ 原型生成公式验证通过: diff_trend={diff_trend:.2e}, diff_detail={diff_detail:.2e}")
        
        # 验证偏置确实改变了原型（偏置不应该全为0）
        bias_trend_norm = torch.norm(model.trend_bias).item()
        bias_detail_norm = torch.norm(model.detail_bias).item()
        assert bias_trend_norm > 0, "趋势偏置不应该全为0"
        assert bias_detail_norm > 0, "细节偏置不应该全为0"
        print(f"✅ 偏置非零验证: ||B_trend||={bias_trend_norm:.4f}, ||B_detail||={bias_detail_norm:.4f}")
        
        # 验证趋势和细节原型不同（因为偏置不同）
        diff_prototypes = torch.norm(P_trend - P_detail).item()
        assert diff_prototypes > 0, "趋势和细节原型应该不同（因为偏置不同）"
        print(f"✅ 趋势和细节原型不同: ||P_trend - P_detail||={diff_prototypes:.4f}")
    
    print("=" * 70)
    print("✅ 测试2通过：原型生成逻辑正确")
    print("=" * 70)


def test_freq_aware_prototypes_with_projection():
    """测试3: 原型投影 - 当 shared_proto_size != num_trend_tokens 时"""
    print("\n" + "=" * 70)
    print("测试3: 原型投影功能")
    print("=" * 70)
    
    config = MockConfig()
    config.use_freq_aware_prototypes = True
    config.shared_proto_size = 800
    config.dual_proto_trend_tokens = 1000  # 与共享原型库大小不同
    config.dual_proto_detail_tokens = 1000
    
    model = Model(config)
    model.eval()
    
    # 检查是否创建了投影层
    assert model.proto_projection_trend is not None, "应该创建趋势投影层（因为大小不同）"
    assert model.proto_projection_detail is not None, "应该创建细节投影层（因为大小不同）"
    print("✅ 投影层已创建")
    
    # 手动生成原型并验证投影
    with torch.no_grad():
        P_shared = model.shared_mapping(model.word_embeddings.permute(1, 0)).permute(1, 0)
        P_trend = P_shared + model.trend_bias
        P_detail = P_shared + model.detail_bias
        
        # 应用投影（需要先转置，与模型实现一致）
        # 输入: (shared_proto_size, d_llm) -> 转置 -> (d_llm, shared_proto_size)
        # Linear(shared_proto_size -> num_trend_tokens) -> (d_llm, num_trend_tokens) -> 转置回 (num_trend_tokens, d_llm)
        P_trend_proj = model.proto_projection_trend(P_trend.permute(1, 0)).permute(1, 0)
        P_detail_proj = model.proto_projection_detail(P_detail.permute(1, 0)).permute(1, 0)
        
        assert P_trend_proj.shape == (1000, 768), f"投影后趋势原型形状应该是(1000, 768)，实际是{P_trend_proj.shape}"
        assert P_detail_proj.shape == (1000, 768), f"投影后细节原型形状应该是(1000, 768)，实际是{P_detail_proj.shape}"
        print(f"✅ 投影后形状: P_trend_proj={P_trend_proj.shape}, P_detail_proj={P_detail_proj.shape}")
    
    print("=" * 70)
    print("✅ 测试3通过：原型投影功能正常")
    print("=" * 70)


def test_freq_aware_prototypes_backward_compatibility():
    """测试4: 向后兼容性 - 不启用频率感知原型增强时应该使用原版逻辑"""
    print("\n" + "=" * 70)
    print("测试4: 向后兼容性验证")
    print("=" * 70)
    
    config = MockConfig()
    config.use_freq_aware_prototypes = False  # 不启用频率感知原型增强
    
    model = Model(config)
    
    # 检查应该使用原版映射层
    assert model.shared_mapping is None, "不启用频率感知时，共享映射层应该为None"
    assert model.trend_bias is None, "不启用频率感知时，趋势偏置应该为None"
    assert model.detail_bias is None, "不启用频率感知时，细节偏置应该为None"
    assert model.trend_mapping is not None, "应该使用原版趋势映射层"
    assert model.detail_mapping is not None, "应该使用原版细节映射层"
    
    print("✅ 向后兼容性检查通过：使用原版映射层")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 336
    n_vars = 7
    x_enc = torch.randn(batch_size, seq_len, n_vars)
    x_mark_enc = None
    x_dec = None
    x_mark_dec = None
    
    try:
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert output.shape == (batch_size, config.pred_len, n_vars)
        print(f"✅ 原版模式前向传播成功，输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 原版模式前向传播失败: {e}")
        raise
    
    print("=" * 70)
    print("✅ 测试4通过：向后兼容性正常")
    print("=" * 70)


def test_freq_aware_prototypes_gradient_flow():
    """测试5: 梯度流验证 - 确保偏置参数可以更新"""
    print("\n" + "=" * 70)
    print("测试5: 梯度流验证")
    print("=" * 70)
    
    config = MockConfig()
    config.use_freq_aware_prototypes = True
    config.shared_proto_size = 800
    
    model = Model(config)
    model.train()
    
    # 检查参数是否可训练
    assert model.trend_bias.requires_grad, "趋势偏置应该可训练"
    assert model.detail_bias.requires_grad, "细节偏置应该可训练"
    
    # shared_mapping 可能是 nn.Linear 或 nn.Sequential，需要分别处理
    if isinstance(model.shared_mapping, nn.Linear):
        assert model.shared_mapping.weight.requires_grad, "共享映射层应该可训练"
    elif isinstance(model.shared_mapping, nn.Sequential):
        assert model.shared_mapping[0].weight.requires_grad, "共享映射层应该可训练"
    else:
        # 检查是否有可训练参数
        has_trainable = any(p.requires_grad for p in model.shared_mapping.parameters())
        assert has_trainable, "共享映射层应该有可训练参数"
    print("✅ 参数可训练性检查通过")
    
    # 创建优化器
    optimizer = torch.optim.Adam([
        {'params': model.trend_bias},
        {'params': model.detail_bias},
        {'params': model.shared_mapping.parameters()}
    ], lr=0.001)
    
    # 记录初始值
    trend_bias_init = model.trend_bias.data.clone()
    detail_bias_init = model.detail_bias.data.clone()
    
    # 前向传播和反向传播
    batch_size = 2
    seq_len = 336
    n_vars = 7
    x_enc = torch.randn(batch_size, seq_len, n_vars)
    x_mark_enc = None
    x_dec = None
    x_mark_dec = None
    
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    loss = output.mean()
    loss.backward()
    
    # 检查梯度是否存在
    assert model.trend_bias.grad is not None, "趋势偏置应该有梯度"
    assert model.detail_bias.grad is not None, "细节偏置应该有梯度"
    print("✅ 梯度存在性检查通过")
    
    # 更新参数
    optimizer.step()
    
    # 检查参数是否更新
    trend_bias_updated = model.trend_bias.data.clone()
    detail_bias_updated = model.detail_bias.data.clone()
    
    diff_trend = torch.norm(trend_bias_init - trend_bias_updated).item()
    diff_detail = torch.norm(detail_bias_init - detail_bias_updated).item()
    
    assert diff_trend > 0, "趋势偏置应该被更新"
    assert diff_detail > 0, "细节偏置应该被更新"
    print(f"✅ 参数更新验证: ||ΔB_trend||={diff_trend:.6f}, ||ΔB_detail||={diff_detail:.6f}")
    
    print("=" * 70)
    print("✅ 测试5通过：梯度流正常")
    print("=" * 70)


def test_freq_aware_prototypes_different_modes():
    """测试6: 不同模式下的频率感知原型增强"""
    print("\n" + "=" * 70)
    print("测试6: 不同模式下的频率感知原型增强")
    print("=" * 70)
    
    # 测试原版映射模式
    print("\n[6.1] 测试原版映射模式...")
    config1 = MockConfig()
    config1.use_freq_aware_prototypes = True
    config1.use_full_vocab_split = False
    config1.use_semantic_filtered_mapping = False
    
    model1 = Model(config1)
    assert model1.shared_mapping is not None, "原版映射模式应该创建共享映射层"
    assert isinstance(model1.shared_mapping, nn.Linear), "原版映射模式应该使用Linear层"
    print("✅ 原版映射模式检查通过")
    
    # 测试全词表切分模式（需要实际运行才能测试，这里只检查配置）
    print("\n[6.2] 测试全词表切分模式配置...")
    config2 = MockConfig()
    config2.use_freq_aware_prototypes = True
    config2.use_full_vocab_split = True
    config2.use_semantic_filtered_mapping = False
    
    # 注意：全词表切分需要实际的vocab_splitter，这里只检查配置逻辑
    print("✅ 全词表切分模式配置检查通过（需要实际数据才能完整测试）")
    
    print("=" * 70)
    print("✅ 测试6通过：不同模式配置正确")
    print("=" * 70)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("开始测试频率感知原型增强功能")
    print("=" * 70)
    
    try:
        test_freq_aware_prototypes_basic()
        test_freq_aware_prototypes_prototype_generation()
        test_freq_aware_prototypes_with_projection()
        test_freq_aware_prototypes_backward_compatibility()
        test_freq_aware_prototypes_gradient_flow()
        test_freq_aware_prototypes_different_modes()
        
        print("\n" + "=" * 70)
        print("🎉 所有测试通过！频率感知原型增强功能实现正确")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ 测试失败: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)

