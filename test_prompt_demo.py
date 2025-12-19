#!/usr/bin/env python3
"""
演示小波特征集成到TimeLLM Prompt的实际效果
展示完整的prompt生成结果
"""

import torch
import numpy as np
from models.TimeLLM import Model
import argparse

def create_demo_config():
    """创建演示配置"""
    class DemoConfig:
        def __init__(self):
            # 基本配置
            self.task_name = 'long_term_forecast'
            self.pred_len = 24
            self.seq_len = 96
            self.d_ff = 32
            self.llm_dim = 768
            self.patch_len = 16
            self.stride = 8
            self.enc_in = 7  # ETT数据集的变量数
            self.dropout = 0.1
            self.d_model = 16
            self.n_heads = 8
            self.llm_layers = 2
            
            # LLM配置
            self.llm_model = 'GPT2'
            self.prompt_domain = 0
            
            # 小波配置
            self.wavelet_mode = 'none'
            self.use_haar_wavelet = 0
            
            # 输出头配置
            self.use_dual_scale_head = 0
            self.use_freq_decoupled_head = 0
    
    return DemoConfig()

def create_realistic_signals():
    """创建更真实的时间序列数据"""
    T = 96
    B = 3  # 3个样本
    N = 7  # 7个变量（模拟ETT数据集）
    
    # 创建模拟的ETT数据
    t = torch.linspace(0, 4*np.pi, T)
    
    signals = torch.zeros(B, T, N)
    
    # 样本1: 平稳的电力负荷数据（低频主导）
    for i in range(N):
        base_trend = 50 + 10 * torch.sin(0.5 * t + i * 0.1)  # 基础趋势
        daily_cycle = 5 * torch.sin(2 * t + i * 0.2)  # 日周期
        noise = 0.5 * torch.randn(T)  # 小噪声
        signals[0, :, i] = base_trend + daily_cycle + noise
    
    # 样本2: 波动的温度数据（中等波动）
    for i in range(N):
        base_temp = 20 + 15 * torch.sin(0.3 * t + i * 0.15)
        weather_var = 3 * torch.sin(4 * t + i * 0.3) * torch.exp(-0.1 * t)
        noise = 1.0 * torch.randn(T)
        signals[1, :, i] = base_temp + weather_var + noise
    
    # 样本3: 高频噪声数据（设备故障场景）
    for i in range(N):
        base_signal = 30 + 2 * t / T  # 轻微趋势
        high_freq_noise = 8 * torch.randn(T)  # 强噪声
        spikes = 15 * (torch.rand(T) > 0.95).float()  # 随机尖峰
        signals[2, :, i] = base_signal + high_freq_noise + spikes
    
    return signals

def demo_full_prompt_generation():
    """演示完整的prompt生成过程"""
    print("🎯 TimeLLM小波Prompt完整演示")
    print("=" * 80)
    
    try:
        # 创建模型
        config = create_demo_config()
        model = Model(config)
        
        # 创建真实的测试数据
        x_enc = create_realistic_signals()  # (B=3, T=96, N=7)
        print(f"输入数据形状: {x_enc.shape}")
        
        # 模拟时间标记（可以是空的）
        x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4)  # 时间特征
        x_dec = torch.zeros(x_enc.shape[0], config.pred_len, x_enc.shape[2])
        x_mark_dec = torch.zeros(x_enc.shape[0], config.pred_len, 4)
        
        # 手动执行forecast函数的前半部分来获取prompt
        print("\n" + "=" * 80)
        print("生成的Prompt示例")
        print("=" * 80)
        
        # 归一化
        x_enc_norm = model.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc_norm.size()
        x_enc_reshaped = x_enc_norm.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # 计算统计量
        min_values = torch.min(x_enc_reshaped, dim=1)[0]
        max_values = torch.max(x_enc_reshaped, dim=1)[0]
        medians = torch.median(x_enc_reshaped, dim=1).values
        lags = model.calcute_lags(x_enc_reshaped)
        trends = x_enc_reshaped.diff(dim=1).sum(dim=1)
        
        # 生成prompt（只显示前几个样本）
        sample_indices = [0, 7, 14]  # 每个batch的第一个变量
        scenario_names = ["平稳电力负荷", "波动温度数据", "高频噪声数据"]
        
        for idx, (sample_idx, scenario) in enumerate(zip(sample_indices, scenario_names)):
            print(f"\n📊 场景 {idx+1}: {scenario}")
            print("-" * 60)
            
            # 格式化统计值
            min_val = min_values[sample_idx].tolist()[0]
            max_val = max_values[sample_idx].tolist()[0]
            median_val = medians[sample_idx].tolist()[0]
            
            min_values_str = f"{min_val:.3f}"
            max_values_str = f"{max_val:.3f}"
            median_values_str = f"{median_val:.3f}"
            lags_values_str = str(lags[sample_idx].tolist())
            
            # 小波特征分析
            current_x = x_enc_reshaped[sample_idx, :, 0]
            hfer, volatility, smoothness_level = model.analyze_wavelet_features(current_x)
            wavelet_desc = model.get_wavelet_description(hfer, volatility, smoothness_level)
            
            # 生成完整prompt
            prompt = (
                f"<|start_prompt|>Dataset description: {model.description}"
                f"Task description: forecast the next {str(model.pred_len)} steps given the previous {str(model.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[sample_idx] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}; "
                f"Frequency characteristics: {wavelet_desc}."
                f"<|<end_prompt>|>"
            )
            
            print("🔤 生成的Prompt:")
            print(prompt)
            
            print(f"\n📈 小波特征详情:")
            print(f"  - 高频能量占比: {hfer:.3f} ({hfer*100:.1f}%)")
            print(f"  - 波动性指标: {volatility:.3f}")
            print(f"  - 平滑度等级: {smoothness_level}/4")
            
        print("\n" + "=" * 80)
        print("✅ 完整演示成功！")
        
        # 对比分析
        print("\n📊 不同场景的小波特征对比:")
        print("-" * 60)
        print(f"{'场景':<12} {'HFER':<8} {'波动性':<8} {'等级':<4} {'LLM理解'}")
        print("-" * 60)
        
        for idx, (sample_idx, scenario) in enumerate(zip(sample_indices, scenario_names)):
            current_x = x_enc_reshaped[sample_idx, :, 0]
            hfer, volatility, smoothness_level = model.analyze_wavelet_features(current_x)
            
            if smoothness_level <= 1:
                llm_hint = "关注趋势预测"
            elif smoothness_level <= 2:
                llm_hint = "平衡趋势与波动"
            else:
                llm_hint = "谨慎处理噪声"
            
            print(f"{scenario:<12} {hfer:<8.3f} {volatility:<8.3f} {smoothness_level:<4} {llm_hint}")
        
    except Exception as e:
        print(f"❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()

def compare_before_after():
    """对比添加小波特征前后的prompt"""
    print("\n" + "=" * 80)
    print("🔄 Prompt对比：添加小波特征前 vs 后")
    print("=" * 80)
    
    # 模拟数据
    min_val, max_val, median_val = 1.234, 5.678, 3.456
    trend = "upward"
    lags = [1, 24, 48, 72, 96]
    
    # 模拟小波特征
    hfer, volatility, smoothness_level = 0.156, 0.423, 2
    
    print("📜 原始Prompt (无小波特征):")
    print("-" * 40)
    original_prompt = (
        f"<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment."
        f"Task description: forecast the next 24 steps given the previous 96 steps information; "
        "Input statistics: "
        f"min value {min_val:.3f}, "
        f"max value {max_val:.3f}, "
        f"median value {median_val:.3f}, "
        f"the trend of input is {trend}, "
        f"top 5 lags are : {lags}"
        f"<|<end_prompt>|>"
    )
    print(original_prompt)
    
    print(f"\n📊 增强Prompt (含小波特征):")
    print("-" * 40)
    wavelet_desc = f"The signal is moderately smooth with some variations with moderate volatility (HF energy: {hfer:.1%})"
    enhanced_prompt = (
        f"<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment."
        f"Task description: forecast the next 24 steps given the previous 96 steps information; "
        "Input statistics: "
        f"min value {min_val:.3f}, "
        f"max value {max_val:.3f}, "
        f"median value {median_val:.3f}, "
        f"the trend of input is {trend}, "
        f"top 5 lags are : {lags}; "
        f"Frequency characteristics: {wavelet_desc}."
        f"<|<end_prompt>|>"
    )
    print(enhanced_prompt)
    
    print(f"\n🎯 关键改进:")
    print("1. ✅ 添加了频域特征描述")
    print("2. ✅ 量化了信号的平滑度和波动性")
    print("3. ✅ 为LLM提供了更丰富的上下文信息")
    print("4. ✅ 帮助LLM理解应该采用保守还是激进的预测策略")

if __name__ == "__main__":
    # 完整演示
    demo_full_prompt_generation()
    
    # 对比分析
    compare_before_after()
    
    print("\n" + "🎉" * 20)
    print("小波特征集成到TimeLLM Prompt的实现已完成！")
    print("🎉" * 20)
