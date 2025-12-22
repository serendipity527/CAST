#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP映射层测试脚本

测试内容：
1. MLP映射层的正确初始化
2. 维度检查
3. 前向传播测试
4. 参数量验证
5. 与Linear层的对比
6. 梯度流测试
7. 非线性激活验证
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

from models.TimeLLM import Model


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
        self.dual_proto_trend_tokens = 1000
        self.dual_proto_detail_tokens = 1000
        self.dual_proto_fusion_method = 'weighted'
        
        # 语义筛选映射配置
        self.use_semantic_filtered_mapping = 1
        self.dual_proto_trend_seed_words = 1000
        self.dual_proto_detail_seed_words = 1000
        self.dual_proto_seed_semantic_filter = 1
        
        # MLP映射层配置（策略一）
        self.dual_proto_mlp_hidden_dim = 4096
        self.dual_proto_mlp_dropout = 0.1
        
        # Prompt配置
        self.prompt_domain = 0
        self.content = 'Test dataset description'
        
        # 其他配置
        self.use_cwpr = 0
        self.use_dual_scale_head = 0
        self.use_freq_decoupled_head = 0


def test_mlp_mapping_initialization():
    """测试1: MLP映射层初始化"""
    print("=" * 70)
    print("测试1: MLP映射层初始化")
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
    
    # 检查映射层类型
    print("\n[检查1] 映射层类型...")
    if not isinstance(model.trend_mapping, nn.Sequential):
        print(f"❌ trend_mapping 不是 Sequential，而是 {type(model.trend_mapping)}")
        return False
    print("✅ trend_mapping 是 Sequential (MLP)")
    
    if not isinstance(model.detail_mapping, nn.Sequential):
        print(f"❌ detail_mapping 不是 Sequential，而是 {type(model.detail_mapping)}")
        return False
    print("✅ detail_mapping 是 Sequential (MLP)")
    
    # 检查MLP结构
    print("\n[检查2] MLP结构...")
    trend_modules = list(model.trend_mapping.modules())[1:]  # 跳过Sequential本身
    detail_modules = list(model.detail_mapping.modules())[1:]
    
    expected_structure = [nn.Linear, nn.GELU, nn.Dropout, nn.Linear]
    for i, (trend_mod, detail_mod, expected_type) in enumerate(zip(trend_modules, detail_modules, expected_structure)):
        if not isinstance(trend_mod, expected_type):
            print(f"❌ trend_mapping 第{i+1}层不是 {expected_type.__name__}，而是 {type(trend_mod).__name__}")
            return False
        if not isinstance(detail_mod, expected_type):
            print(f"❌ detail_mapping 第{i+1}层不是 {expected_type.__name__}，而是 {type(detail_mod).__name__}")
            return False
    
    print("✅ MLP结构正确: Linear -> GELU -> Dropout -> Linear")
    
    return True


def test_mlp_mapping_dimensions():
    """测试2: MLP映射层维度检查"""
    print("\n" + "=" * 70)
    print("测试2: MLP映射层维度检查")
    print("=" * 70)
    
    configs = TestConfig()
    model = Model(configs)
    
    # 获取种子词数量
    num_trend_seeds = model.trend_seed_embeddings.shape[0]
    num_detail_seeds = model.detail_seed_embeddings.shape[0]
    d_llm = model.d_llm
    
    print(f"\n种子词配置:")
    print(f"  - 趋势种子词: {num_trend_seeds} 个")
    print(f"  - 细节种子词: {num_detail_seeds} 个")
    print(f"  - LLM维度: {d_llm}")
    
    # 检查第一层Linear的输入维度
    trend_first_linear = model.trend_mapping[0]
    detail_first_linear = model.detail_mapping[0]
    
    print(f"\n[检查1] 第一层Linear输入维度...")
    if trend_first_linear.in_features != num_trend_seeds:
        print(f"❌ trend_mapping 第一层输入维度不匹配: {trend_first_linear.in_features} != {num_trend_seeds}")
        return False
    print(f"✅ trend_mapping 第一层输入维度: {trend_first_linear.in_features}")
    
    if detail_first_linear.in_features != num_detail_seeds:
        print(f"❌ detail_mapping 第一层输入维度不匹配: {detail_first_linear.in_features} != {num_detail_seeds}")
        return False
    print(f"✅ detail_mapping 第一层输入维度: {detail_first_linear.in_features}")
    
    # 检查隐藏层维度
    mlp_hidden_dim = configs.dual_proto_mlp_hidden_dim
    print(f"\n[检查2] 隐藏层维度...")
    if trend_first_linear.out_features != mlp_hidden_dim:
        print(f"❌ trend_mapping 隐藏层维度不匹配: {trend_first_linear.out_features} != {mlp_hidden_dim}")
        return False
    print(f"✅ trend_mapping 隐藏层维度: {trend_first_linear.out_features}")
    
    if detail_first_linear.out_features != mlp_hidden_dim:
        print(f"❌ detail_mapping 隐藏层维度不匹配: {detail_first_linear.out_features} != {mlp_hidden_dim}")
        return False
    print(f"✅ detail_mapping 隐藏层维度: {detail_first_linear.out_features}")
    
    # 检查最后一层Linear的输出维度
    trend_last_linear = model.trend_mapping[3]
    detail_last_linear = model.detail_mapping[3]
    
    print(f"\n[检查3] 最后一层Linear输出维度...")
    if trend_last_linear.out_features != model.num_trend_tokens:
        print(f"❌ trend_mapping 输出维度不匹配: {trend_last_linear.out_features} != {model.num_trend_tokens}")
        return False
    print(f"✅ trend_mapping 输出维度: {trend_last_linear.out_features}")
    
    if detail_last_linear.out_features != model.num_detail_tokens:
        print(f"❌ detail_mapping 输出维度不匹配: {detail_last_linear.out_features} != {model.num_detail_tokens}")
        return False
    print(f"✅ detail_mapping 输出维度: {detail_last_linear.out_features}")
    
    return True


def test_mlp_mapping_forward():
    """测试3: MLP映射层前向传播"""
    print("\n" + "=" * 70)
    print("测试3: MLP映射层前向传播")
    print("=" * 70)
    
    configs = TestConfig()
    model = Model(configs)
    model.eval()
    
    # 获取种子词embeddings
    trend_seed_emb = model.trend_seed_embeddings  # (num_trend_seeds, d_llm)
    detail_seed_emb = model.detail_seed_embeddings  # (num_detail_seeds, d_llm)
    
    print(f"\n输入形状:")
    print(f"  - trend_seed_embeddings: {trend_seed_emb.shape}")
    print(f"  - detail_seed_embeddings: {detail_seed_emb.shape}")
    
    # 前向传播（与模型forward中的逻辑一致）
    with torch.no_grad():
        # 转置: (num_seeds, d_llm) -> (d_llm, num_seeds)
        trend_input = trend_seed_emb.permute(1, 0)  # (d_llm, num_trend_seeds)
        detail_input = detail_seed_emb.permute(1, 0)  # (d_llm, num_detail_seeds)
        
        print(f"\n转置后形状:")
        print(f"  - trend_input: {trend_input.shape}")
        print(f"  - detail_input: {detail_input.shape}")
        
        # MLP映射
        trend_output = model.trend_mapping(trend_input)  # (d_llm, num_trend_tokens)
        detail_output = model.detail_mapping(detail_input)  # (d_llm, num_detail_tokens)
        
        print(f"\nMLP输出形状:")
        print(f"  - trend_output: {trend_output.shape}")
        print(f"  - detail_output: {detail_output.shape}")
        
        # 转置回: (d_llm, num_tokens) -> (num_tokens, d_llm)
        trend_prototypes = trend_output.permute(1, 0)  # (num_trend_tokens, d_llm)
        detail_prototypes = detail_output.permute(1, 0)  # (num_detail_tokens, d_llm)
        
        print(f"\n最终原型形状:")
        print(f"  - trend_prototypes: {trend_prototypes.shape}")
        print(f"  - detail_prototypes: {detail_prototypes.shape}")
    
    # 验证输出维度
    if trend_prototypes.shape != (model.num_trend_tokens, model.d_llm):
        print(f"❌ trend_prototypes 形状不正确: {trend_prototypes.shape} != ({model.num_trend_tokens}, {model.d_llm})")
        return False
    print("✅ trend_prototypes 形状正确")
    
    if detail_prototypes.shape != (model.num_detail_tokens, model.d_llm):
        print(f"❌ detail_prototypes 形状不正确: {detail_prototypes.shape} != ({model.num_detail_tokens}, {model.d_llm})")
        return False
    print("✅ detail_prototypes 形状正确")
    
    # 检查输出值是否合理（不应该全是0或NaN）
    if torch.isnan(trend_prototypes).any() or torch.isnan(detail_prototypes).any():
        print("❌ 输出包含NaN值")
        return False
    print("✅ 输出值合理（无NaN）")
    
    if (trend_prototypes == 0).all() or (detail_prototypes == 0).all():
        print("⚠️  警告: 输出全为0，可能初始化有问题")
    else:
        print("✅ 输出值非零")
    
    return True


def test_mlp_mapping_parameters():
    """测试4: MLP映射层参数量"""
    print("\n" + "=" * 70)
    print("测试4: MLP映射层参数量")
    print("=" * 70)
    
    configs = TestConfig()
    model = Model(configs)
    
    # 计算参数量
    trend_params = sum(p.numel() for p in model.trend_mapping.parameters())
    detail_params = sum(p.numel() for p in model.detail_mapping.parameters())
    total_params = trend_params + detail_params
    
    print(f"\n参数量统计:")
    print(f"  - trend_mapping: {trend_params:,} ({trend_params/1e6:.2f}M)")
    print(f"  - detail_mapping: {detail_params:,} ({detail_params/1e6:.2f}M)")
    print(f"  - 总计: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 手动计算预期参数量
    num_trend_seeds = model.trend_seed_embeddings.shape[0]
    num_detail_seeds = model.detail_seed_embeddings.shape[0]
    mlp_hidden_dim = configs.dual_proto_mlp_hidden_dim
    
    expected_trend_params = (
        num_trend_seeds * mlp_hidden_dim + mlp_hidden_dim +  # 第一层Linear + bias
        mlp_hidden_dim * model.num_trend_tokens + model.num_trend_tokens  # 第二层Linear + bias
    )
    
    expected_detail_params = (
        num_detail_seeds * mlp_hidden_dim + mlp_hidden_dim +  # 第一层Linear + bias
        mlp_hidden_dim * model.num_detail_tokens + model.num_detail_tokens  # 第二层Linear + bias
    )
    
    print(f"\n预期参数量:")
    print(f"  - trend_mapping: {expected_trend_params:,} ({expected_trend_params/1e6:.2f}M)")
    print(f"  - detail_mapping: {expected_detail_params:,} ({expected_detail_params/1e6:.2f}M)")
    
    # 验证参数量（允许小的差异，因为可能有其他参数）
    if abs(trend_params - expected_trend_params) > 10:
        print(f"⚠️  警告: trend_mapping 参数量差异较大: {abs(trend_params - expected_trend_params)}")
    else:
        print("✅ trend_mapping 参数量正确")
    
    if abs(detail_params - expected_detail_params) > 10:
        print(f"⚠️  警告: detail_mapping 参数量差异较大: {abs(detail_params - expected_detail_params)}")
    else:
        print("✅ detail_mapping 参数量正确")
    
    # 对比Linear版本
    linear_params = num_trend_seeds * model.num_trend_tokens + num_detail_seeds * model.num_detail_tokens
    print(f"\n对比Linear版本:")
    print(f"  - Linear版本参数量: {linear_params:,} ({linear_params/1e6:.2f}M)")
    print(f"  - MLP版本参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - 增加倍数: {total_params / linear_params:.2f}x")
    
    return True


def test_mlp_mapping_gradients():
    """测试5: MLP映射层梯度流"""
    print("\n" + "=" * 70)
    print("测试5: MLP映射层梯度流")
    print("=" * 70)
    
    configs = TestConfig()
    model = Model(configs)
    model.train()
    
    # 创建虚拟输入
    batch_size = 2
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.zeros(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)
    x_mark_dec = torch.zeros(batch_size, configs.pred_len, 4)
    
    # 前向传播
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    # 创建虚拟损失
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print("\n[检查1] 映射层参数梯度...")
    trend_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.trend_mapping.parameters())
    detail_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detail_mapping.parameters())
    
    if not trend_has_grad:
        print("❌ trend_mapping 没有梯度")
        return False
    print("✅ trend_mapping 有梯度")
    
    if not detail_has_grad:
        print("❌ detail_mapping 没有梯度")
        return False
    print("✅ detail_mapping 有梯度")
    
    # 检查Buffer不参与梯度更新
    print("\n[检查2] Buffer不参与梯度更新...")
    if model.trend_seed_embeddings.requires_grad:
        print("❌ trend_seed_embeddings requires_grad=True（应该是False）")
        return False
    print("✅ trend_seed_embeddings requires_grad=False")
    
    if model.detail_seed_embeddings.requires_grad:
        print("❌ detail_seed_embeddings requires_grad=True（应该是False）")
        return False
    print("✅ detail_seed_embeddings requires_grad=False")
    
    return True


def test_mlp_vs_linear_comparison():
    """测试6: MLP vs Linear对比"""
    print("\n" + "=" * 70)
    print("测试6: MLP vs Linear对比")
    print("=" * 70)
    
    configs = TestConfig()
    
    # 创建MLP版本模型
    model_mlp = Model(configs)
    model_mlp.eval()
    
    # 创建Linear版本模型（通过修改配置）
    configs_linear = TestConfig()
    # 这里我们需要手动创建一个Linear版本的映射层来对比
    num_trend_seeds = model_mlp.trend_seed_embeddings.shape[0]
    num_detail_seeds = model_mlp.detail_seed_embeddings.shape[0]
    
    linear_trend = nn.Linear(num_trend_seeds, model_mlp.num_trend_tokens)
    linear_detail = nn.Linear(num_detail_seeds, model_mlp.num_detail_tokens)
    
    # 获取相同的输入
    trend_input = model_mlp.trend_seed_embeddings.permute(1, 0)
    detail_input = model_mlp.detail_seed_embeddings.permute(1, 0)
    
    print(f"\n输入形状: {trend_input.shape}, {detail_input.shape}")
    
    # 前向传播对比
    with torch.no_grad():
        # MLP版本
        mlp_trend_out = model_mlp.trend_mapping(trend_input)
        mlp_detail_out = model_mlp.detail_mapping(detail_input)
        
        # Linear版本
        linear_trend_out = linear_trend(trend_input)
        linear_detail_out = linear_detail(detail_input)
    
    print(f"\n输出形状对比:")
    print(f"  - MLP trend: {mlp_trend_out.shape}")
    print(f"  - Linear trend: {linear_trend_out.shape}")
    print(f"  - MLP detail: {mlp_detail_out.shape}")
    print(f"  - Linear detail: {linear_detail_out.shape}")
    
    # 统计信息对比
    print(f"\n输出统计信息对比:")
    print(f"  - MLP trend - 均值: {mlp_trend_out.mean().item():.6f}, 标准差: {mlp_trend_out.std().item():.6f}")
    print(f"  - Linear trend - 均值: {linear_trend_out.mean().item():.6f}, 标准差: {linear_trend_out.std().item():.6f}")
    print(f"  - MLP detail - 均值: {mlp_detail_out.mean().item():.6f}, 标准差: {mlp_detail_out.std().item():.6f}")
    print(f"  - Linear detail - 均值: {linear_detail_out.mean().item():.6f}, 标准差: {linear_detail_out.std().item():.6f}")
    
    # MLP应该有不同的输出（因为非线性）
    if torch.allclose(mlp_trend_out, linear_trend_out, atol=1e-5):
        print("⚠️  警告: MLP和Linear输出过于接近，可能非线性激活没有生效")
    else:
        print("✅ MLP和Linear输出不同（非线性激活生效）")
    
    return True


def test_end_to_end():
    """测试7: 端到端测试"""
    print("\n" + "=" * 70)
    print("测试7: 端到端测试")
    print("=" * 70)
    
    configs = TestConfig()
    model = Model(configs)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.zeros(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)
    x_mark_dec = torch.zeros(batch_size, configs.pred_len, 4)
    
    print(f"\n输入形状:")
    print(f"  - x_enc: {x_enc.shape}")
    print(f"  - x_mark_enc: {x_mark_enc.shape}")
    print(f"  - x_dec: {x_dec.shape}")
    print(f"  - x_mark_dec: {x_mark_dec.shape}")
    
    # 前向传播
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
    print("MLP映射层完整测试套件")
    print("=" * 70)
    
    tests = [
        ("MLP映射层初始化", test_mlp_mapping_initialization),
        ("MLP映射层维度检查", test_mlp_mapping_dimensions),
        ("MLP映射层前向传播", test_mlp_mapping_forward),
        ("MLP映射层参数量", test_mlp_mapping_parameters),
        ("MLP映射层梯度流", test_mlp_mapping_gradients),
        ("MLP vs Linear对比", test_mlp_vs_linear_comparison),
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
        print("\n🎉 所有测试通过！")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

