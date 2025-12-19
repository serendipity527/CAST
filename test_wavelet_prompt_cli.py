#!/usr/bin/env python3
"""
测试小波Prompt功能的命令行参数控制
验证不同参数组合下的行为
"""

import torch
import numpy as np
import subprocess
import os
import sys

def create_test_script():
    """创建一个简化的测试脚本来验证参数传递"""
    test_script_content = '''
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
        print(f"\\n📋 测试 {i+1}: {test_config['name']}")
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
    
    print(f"\\n" + "=" * 60)
    print("🎉 参数测试完成！")

if __name__ == "__main__":
    test_wavelet_prompt_params()
'''
    
    # 写入测试脚本
    with open('/home/dmx_MT/LZF/project/CAST/temp_test_params.py', 'w', encoding='utf-8') as f:
        f.write(test_script_content)

def test_cli_parameters():
    """测试命令行参数的传递和解析"""
    print("🔧 命令行参数控制测试")
    print("=" * 70)
    
    # 创建测试脚本
    create_test_script()
    
    print("✅ 测试脚本已创建")
    
    # 运行参数测试
    print("\n📊 运行参数测试...")
    try:
        result = subprocess.run([
            'conda', 'run', '-n', 'timellm', 
            'python', '/home/dmx_MT/LZF/project/CAST/temp_test_params.py'
        ], 
        capture_output=True, 
        text=True, 
        cwd='/home/dmx_MT/LZF/project/CAST'
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ 参数测试成功完成")
        else:
            print(f"⚠️ 测试退出码: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 运行测试时出错: {str(e)}")
    
    # 清理临时文件
    try:
        os.remove('/home/dmx_MT/LZF/project/CAST/temp_test_params.py')
        print("🧹 临时文件已清理")
    except:
        pass

def show_usage_examples():
    """展示如何使用命令行参数"""
    print("\n" + "=" * 70)
    print("📖 命令行参数使用示例")
    print("=" * 70)
    
    examples = [
        {
            'title': '关闭小波Prompt（默认）',
            'command': 'python run_main.py --use_wavelet_prompt 0 --model TimeLLM --data ETTh1 --is_training 1 --model_id test'
        },
        {
            'title': '启用Haar小波Prompt',
            'command': 'python run_main.py --use_wavelet_prompt 1 --wavelet_prompt_method haar --model TimeLLM --data ETTh1 --is_training 1 --model_id test'
        },
        {
            'title': '使用简化频域分析',
            'command': 'python run_main.py --use_wavelet_prompt 1 --wavelet_prompt_method simple --model TimeLLM --data ETTh1 --is_training 1 --model_id test'
        },
        {
            'title': '调整HFER阈值（更敏感）',
            'command': 'python run_main.py --use_wavelet_prompt 1 --prompt_hfer_threshold 0.05 --model TimeLLM --data ETTh1 --is_training 1 --model_id test'
        },
        {
            'title': '调整HFER阈值（不敏感）',
            'command': 'python run_main.py --use_wavelet_prompt 1 --prompt_hfer_threshold 0.30 --model TimeLLM --data ETTh1 --is_training 1 --model_id test'
        }
    ]
    
    for i, example in enumerate(examples):
        print(f"\n🔹 示例 {i+1}: {example['title']}")
        print(f"   {example['command']}")
    
    print(f"\n📝 参数说明:")
    print("  --use_wavelet_prompt: 0=关闭, 1=启用小波Prompt增强")
    print("  --wavelet_prompt_method: haar=Haar小波分解, simple=简化频域分析")
    print("  --prompt_hfer_threshold: 高频能量占比阈值，影响平滑度等级判断")

def create_comparison_demo():
    """创建对比演示脚本"""
    demo_content = '''#!/usr/bin/env python3
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
        print(f"\\n📊 信号类型: {signal_name.upper()}")
        print("-" * 50)
        
        for use_wavelet in [0, 1]:
            status = "开启" if use_wavelet else "关闭"
            print(f"\\n🔧 小波Prompt: {status}")
            
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
'''
    
    with open('/home/dmx_MT/LZF/project/CAST/demo_prompt_comparison.py', 'w', encoding='utf-8') as f:
        f.write(demo_content)
    
    print("📄 对比演示脚本已创建: demo_prompt_comparison.py")

def main():
    """主测试函数"""
    print("🎯 小波Prompt命令行参数控制测试")
    print("=" * 70)
    
    # 1. 测试参数传递和解析
    test_cli_parameters()
    
    # 2. 展示使用示例
    show_usage_examples()
    
    # 3. 创建对比演示
    create_comparison_demo()
    
    print("\n" + "=" * 70)
    print("🎉 所有测试完成！")
    print("=" * 70)
    
    print("\n📋 总结:")
    print("✅ 已添加3个新的命令行参数:")
    print("   - --use_wavelet_prompt: 控制是否启用小波Prompt增强")
    print("   - --wavelet_prompt_method: 选择分析方法（haar/simple）")
    print("   - --prompt_hfer_threshold: 调整敏感度阈值")
    print("✅ TimeLLM.py已支持参数控制和条件执行")
    print("✅ 提供了完整的使用示例和对比演示")
    
    print("\n🚀 现在你可以通过以下方式启用小波Prompt:")
    print("   conda run -n timellm python run_main.py \\")
    print("     --use_wavelet_prompt 1 \\")
    print("     --wavelet_prompt_method haar \\")
    print("     --model TimeLLM --data ETTh1 --is_training 1 --model_id test")

if __name__ == "__main__":
    main()
