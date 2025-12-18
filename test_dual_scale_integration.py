"""
测试 DualScaleResidualHead 在 TimeLLM 中的集成
验证模型能否正确加载和使用新的输出头
"""

import torch
import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.TimeLLM import Model


def create_test_config():
    """创建测试配置"""
    class Config:
        def __init__(self):
            # 基本配置
            self.task_name = 'long_term_forecast'
            self.seq_len = 512
            self.pred_len = 96
            self.label_len = 48
            self.enc_in = 7
            self.dec_in = 7
            self.c_out = 7
            
            # 模型配置
            self.d_model = 64
            self.d_ff = 256
            self.n_heads = 8
            self.dropout = 0.1
            
            # LLM 配置
            self.llm_model = 'GPT2'
            self.llm_dim = 768
            self.llm_layers = 6
            
            # Patch 配置
            self.patch_len = 16
            self.stride = 8
            
            # 小波配置
            self.wavelet_mode = 'wist'
            self.wavelet_type = 'haar'
            self.wavelet_level = 2
            self.hf_dropout = 0.2
            self.mf_dropout = 0.2
            self.use_freq_attention = 1
            self.use_soft_threshold = 1
            self.pyramid_fusion = 1
            
            # 输出头配置 - 测试三种模式
            self.use_dual_scale_head = 0  # 将在测试中修改
            self.detail_dropout = 0.1
            
            self.use_freq_decoupled_head = 0  # 将在测试中修改
            self.mid_dropout = 0.2
            self.high_dropout = 0.5
            self.head_soft_threshold = 1
            self.head_soft_threshold_init = 0.1
            self.head_use_conv = 0
            self.use_deep_supervision = 0
            
            # 其他
            self.prompt_domain = 0
            self.content = "Test content"
    
    return Config()


def test_model_creation_and_forward():
    """测试模型创建和前向传播"""
    print("=" * 80)
    print("TimeLLM + DualScaleResidualHead 集成测试")
    print("=" * 80)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 测试参数
    B = 2  # 小 batch 避免内存问题
    
    # ========== 测试 1: FlattenHead (原版) ==========
    print("\n" + "=" * 60)
    print("测试 1: FlattenHead (原版)")
    print("=" * 60)
    
    config = create_test_config()
    config.use_dual_scale_head = 0
    config.use_freq_decoupled_head = 0
    
    try:
        model_flatten = Model(config).to(device)
        
        # 模拟输入
        x_enc = torch.randn(B, config.seq_len, config.enc_in, device=device)
        x_mark_enc = torch.randn(B, config.seq_len, 4, device=device)  # 时间特征
        x_dec = torch.randn(B, config.label_len + config.pred_len, config.dec_in, device=device)
        x_mark_dec = torch.randn(B, config.label_len + config.pred_len, 4, device=device)
        
        # 前向传播
        with torch.no_grad():
            output_flatten = model_flatten(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ FlattenHead 输出形状: {output_flatten.shape}")
        expected_shape = (B, config.pred_len, config.c_out)
        assert output_flatten.shape == expected_shape, f"输出形状错误: {output_flatten.shape}"
        
        flatten_params = sum(p.numel() for p in model_flatten.parameters())
        print(f"✅ FlattenHead 总参数: {flatten_params:,}")
        
    except Exception as e:
        print(f"❌ FlattenHead 测试失败: {e}")
        return False
    
    # ========== 测试 2: DualScaleResidualHead ==========
    print("\n" + "=" * 60)
    print("测试 2: DualScaleResidualHead")
    print("=" * 60)
    
    config = create_test_config()
    config.use_dual_scale_head = 1
    config.use_freq_decoupled_head = 0
    
    try:
        model_dual = Model(config).to(device)
        
        # 前向传播
        with torch.no_grad():
            output_dual = model_dual(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ DualScaleHead 输出形状: {output_dual.shape}")
        assert output_dual.shape == expected_shape, f"输出形状错误: {output_dual.shape}"
        
        dual_params = sum(p.numel() for p in model_dual.parameters())
        print(f"✅ DualScaleHead 总参数: {dual_params:,}")
        print(f"✅ 参数增加: {dual_params - flatten_params:,} ({(dual_params/flatten_params-1)*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ DualScaleHead 测试失败: {e}")
        return False
    
    # ========== 测试 3: TriBandDecoupledHead (对比) ==========
    print("\n" + "=" * 60)
    print("测试 3: TriBandDecoupledHead (对比)")
    print("=" * 60)
    
    config = create_test_config()
    config.use_dual_scale_head = 0
    config.use_freq_decoupled_head = 1
    
    try:
        model_triband = Model(config).to(device)
        
        # 前向传播
        with torch.no_grad():
            output_triband = model_triband(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ TriBandHead 输出形状: {output_triband.shape}")
        assert output_triband.shape == expected_shape, f"输出形状错误: {output_triband.shape}"
        
        triband_params = sum(p.numel() for p in model_triband.parameters())
        print(f"✅ TriBandHead 总参数: {triband_params:,}")
        print(f"✅ 相比 Flatten: {triband_params - flatten_params:,} ({(triband_params/flatten_params-1)*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ TriBandHead 测试失败: {e}")
        return False
    
    # ========== 测试 4: 优先级验证 ==========
    print("\n" + "=" * 60)
    print("测试 4: 优先级验证 (DualScale 优于 TriBand)")
    print("=" * 60)
    
    config = create_test_config()
    config.use_dual_scale_head = 1  # 同时开启两个
    config.use_freq_decoupled_head = 1
    
    try:
        model_priority = Model(config).to(device)
        
        # 检查实际使用的输出头类型
        output_head_type = type(model_priority.output_projection).__name__
        print(f"✅ 同时开启时使用: {output_head_type}")
        assert output_head_type == 'DualScaleResidualHead', f"优先级错误，应该使用 DualScaleResidualHead"
        
        # 前向传播
        with torch.no_grad():
            output_priority = model_priority(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ 优先级测试输出形状: {output_priority.shape}")
        assert output_priority.shape == expected_shape
        
    except Exception as e:
        print(f"❌ 优先级测试失败: {e}")
        return False
    
    # ========== 测试 5: 内存和性能对比 ==========
    print("\n" + "=" * 60)
    print("测试 5: 内存和性能对比")
    print("=" * 60)
    
    def benchmark_model(model, name, x_enc, x_mark_enc, x_dec, x_mark_dec, rounds=10):
        """简单的性能基准测试"""
        import time
        
        model.eval()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 计时
        start_time = time.time()
        with torch.no_grad():
            for _ in range(rounds):
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / rounds
        return avg_time, output
    
    flatten_time, _ = benchmark_model(model_flatten, "FlattenHead", x_enc, x_mark_enc, x_dec, x_mark_dec)
    dual_time, _ = benchmark_model(model_dual, "DualScaleHead", x_enc, x_mark_enc, x_dec, x_mark_dec)
    triband_time, _ = benchmark_model(model_triband, "TriBandHead", x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"⏱️  FlattenHead 平均推理时间: {flatten_time:.4f}s")
    print(f"⏱️  DualScaleHead 平均推理时间: {dual_time:.4f}s (相对: {dual_time/flatten_time:.2f}x)")
    print(f"⏱️  TriBandHead 平均推理时间: {triband_time:.4f}s (相对: {triband_time/flatten_time:.2f}x)")
    
    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("📊 集成测试总结")
    print("=" * 80)
    print(f"✅ 所有输出头类型工作正常")
    print(f"✅ 输出形状一致: {expected_shape}")
    print(f"✅ 优先级逻辑正确: DualScale > TriBand > Flatten")
    print(f"✅ 参数量对比:")
    print(f"   - FlattenHead: {flatten_params:,}")
    print(f"   - DualScaleHead: {dual_params:,} (+{(dual_params/flatten_params-1)*100:.1f}%)")
    print(f"   - TriBandHead: {triband_params:,} (+{(triband_params/flatten_params-1)*100:.1f}%)")
    print(f"✅ 性能对比 (相对 FlattenHead):")
    print(f"   - DualScaleHead: {dual_time/flatten_time:.2f}x")
    print(f"   - TriBandHead: {triband_time/flatten_time:.2f}x")
    
    return True


def main():
    """主函数"""
    success = test_model_creation_and_forward()
    
    if success:
        print("\n🎉 TimeLLM + DualScaleResidualHead 集成成功！")
        print("\n📋 现在可以使用以下命令测试实际训练:")
        print("HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python run_main.py \\")
        print("  --task_name long_term_forecast --is_training 1 \\")
        print("  --root_path ./dataset/ETT-small --data_path ETTh1.csv \\")
        print("  --model_id ETTh1_512_96 --model TimeLLM --data ETTh1 --features M \\")
        print("  --seq_len 512 --label_len 48 --pred_len 96 \\")
        print("  --d_model 64 --d_ff 256 --batch_size 24 --learning_rate 0.0001 \\")
        print("  --llm_model GPT2 --llm_dim 768 --llm_layers 6 --train_epochs 15 \\")
        print("  --wavelet_mode=wist --wavelet_type=haar --wavelet_level=2 \\")
        print("  --use_dual_scale_head=1 --detail_dropout=0.1 \\")
        print("  --model_comment 'WIST-PE-haar-DualScaleHead'")
        print("\n" + "=" * 80)
    else:
        print("\n❌ 集成测试失败，请检查代码")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
