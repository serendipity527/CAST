
import argparse
import torch
import numpy as np
from models.TimeLLM import Model

def create_test_config(args_dict):
    """根据参数字典创建配置对象"""
    class TestConfig:
        def __init__(self, **kwargs):
            # 基本配置
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
            
            # 小波配置
            self.wavelet_mode = 'none'
            self.use_haar_wavelet = 0
            self.use_dual_scale_head = 0
            self.use_freq_decoupled_head = 0
            
            # 从参数字典更新配置
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    return TestConfig(**args_dict)

def test_wavelet_prompt_params():
    """测试不同的小波prompt参数组合"""
    
    # 创建测试数据
    T = 96
    t = torch.linspace(0, 4*np.pi, T)
    
    # 测试信号：混合信号（中等波动）
    test_signal = torch.sin(t) + 0.3 * torch.sin(5 * t) + 0.1 * torch.randn(T)
    x_test = test_signal.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    
    print("🧪 小波Prompt参数测试")
    print("=" * 60)
    
    # 测试配置列表
    test_configs = [
        {
            'name': '关闭小波Prompt',
            'params': {
                'use_wavelet_prompt': 0
            }
        },
        {
            'name': '启用Haar小波Prompt',
            'params': {
                'use_wavelet_prompt': 1,
                'wavelet_prompt_method': 'haar',
                'prompt_hfer_threshold': 0.15
            }
        },
        {
            'name': '启用简化频域分析',
            'params': {
                'use_wavelet_prompt': 1,
                'wavelet_prompt_method': 'simple',
                'prompt_hfer_threshold': 0.15
            }
        },
        {
            'name': '调整HFER阈值（敏感）',
            'params': {
                'use_wavelet_prompt': 1,
                'wavelet_prompt_method': 'haar',
                'prompt_hfer_threshold': 0.05  # 更敏感的阈值
            }
        },
        {
            'name': '调整HFER阈值（不敏感）',
            'params': {
                'use_wavelet_prompt': 1,
                'wavelet_prompt_method': 'haar',
                'prompt_hfer_threshold': 0.30  # 不太敏感的阈值
            }
        }
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\n📋 测试 {i+1}: {test_config['name']}")
        print("-" * 40)
        
        try:
            # 创建配置
            config = create_test_config(test_config['params'])
            
            # 创建模型
            model = Model(config)
            
            # 测试小波特征分析（如果启用）
            if config.use_wavelet_prompt:
                hfer, volatility, smoothness_level = model.analyze_wavelet_features(test_signal)
                wavelet_desc = model.get_wavelet_description(hfer, volatility, smoothness_level)
                
                print(f"  ✅ 小波特征分析成功")
                print(f"     - 方法: {config.wavelet_prompt_method}")
                print(f"     - HFER阈值: {config.prompt_hfer_threshold}")
                print(f"     - 高频能量占比: {hfer:.3f}")
                print(f"     - 波动性: {volatility:.3f}")
                print(f"     - 平滑度等级: {smoothness_level}/4")
                print(f"     - 描述: {wavelet_desc}")
            else:
                print(f"  ✅ 小波Prompt已关闭，使用原版Prompt")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {str(e)}")
    
    print(f"\n" + "=" * 60)
    print("🎉 参数测试完成！")

if __name__ == "__main__":
    test_wavelet_prompt_params()
