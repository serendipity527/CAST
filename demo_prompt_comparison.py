#!/usr/bin/env python3
"""
对比演示：启用/关闭小波Prompt的效果
"""

import argparse
import torch
import numpy as np
from models.TimeLLM import Model

def demo_prompt_comparison():
    """演示启用和关闭小波Prompt的对比效果"""
    
    # 创建测试信号
    T = 96
    t = torch.linspace(0, 4*np.pi, T)
    noisy_signal = torch.randn(T) * 0.8 + torch.sin(t)  # 高噪声信号
    smooth_signal = torch.sin(0.5 * t) + 0.05 * t  # 平滑信号
    
    signals = {
        'noisy': noisy_signal,
        'smooth': smooth_signal
    }
    
    print("🔄 小波Prompt开启/关闭对比演示")
    print("=" * 70)
    
    for signal_name, signal in signals.items():
        print(f"\n📊 信号类型: {signal_name.upper()}")
        print("-" * 50)
        
        for use_wavelet in [0, 1]:
            status = "开启" if use_wavelet else "关闭"
            print(f"\n🔧 小波Prompt: {status}")
            
            # 配置类
            class DemoConfig:
                def __init__(self):
                    self.task_name = 'long_term_forecast'
                    self.pred_len = 24
                    self.seq_len = 96
                    self.d_ff = 32
                    self.llm_dim = 768
                    self.patch_len = 16
                    self.stride = 8
                    self.enc_in = 1
                    self.dropout = 0.1
                    self.d_model = 16
                    self.n_heads = 8
                    self.llm_layers = 2
                    self.llm_model = 'GPT2'
                    self.prompt_domain = 0
                    self.wavelet_mode = 'none'
                    self.use_haar_wavelet = 0
                    self.use_dual_scale_head = 0
                    self.use_freq_decoupled_head = 0
                    
                    # 小波Prompt配置
                    self.use_wavelet_prompt = use_wavelet
                    self.wavelet_prompt_method = 'haar'
                    self.prompt_hfer_threshold = 0.15
            
            try:
                config = DemoConfig()
                model = Model(config)
                
                if use_wavelet:
                    hfer, volatility, smoothness_level = model.analyze_wavelet_features(signal)
                    wavelet_desc = model.get_wavelet_description(hfer, volatility, smoothness_level)
                    print(f"   小波特征: HFER={hfer:.3f}, 波动性={volatility:.3f}, 等级={smoothness_level}")
                    print(f"   描述: {wavelet_desc}")
                else:
                    print(f"   使用原版Prompt（无频域信息）")
                    
            except Exception as e:
                print(f"   ❌ 错误: {str(e)}")

if __name__ == "__main__":
    demo_prompt_comparison()
