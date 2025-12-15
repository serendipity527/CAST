"""
测试 WISTPatchEmbedding 的分层金字塔融合功能
验证点:
1. 模块能正确初始化 (level=1 双通道模式, level>=2 金字塔模式)
2. 前向传播输出维度正确
3. 金字塔融合的门控数量正确
4. 不同配置下的梯度能正常回传
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import unittest


class TestWISTPatchEmbedding(unittest.TestCase):
    """测试 WISTPatchEmbedding 模块"""
    
    @classmethod
    def setUpClass(cls):
        """导入模块"""
        from layers.Embed import WISTPatchEmbedding
        cls.WISTPatchEmbedding = WISTPatchEmbedding
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {cls.device}")
    
    def test_level1_dual_channel_mode(self):
        """测试 level=1 时使用双通道模式"""
        print("\n" + "="*60)
        print("测试 1: level=1 双通道模式")
        print("="*60)
        
        model = self.WISTPatchEmbedding(
            d_model=32,
            patch_len=16,
            stride=8,
            dropout=0.1,
            wavelet_type='haar',
            wavelet_level=1,
            hf_dropout=0.5,
            gate_bias_init=2.0,
            use_soft_threshold=True,
            use_causal_conv=True,
            pyramid_fusion=True,  # 即使设置为 True，level=1 时也应该用双通道
            mf_dropout=0.3
        ).to(self.device)
        
        # 验证: level=1 时不应该启用金字塔融合
        self.assertFalse(model.pyramid_fusion, "level=1 时不应启用金字塔融合")
        
        # 验证: 应该有双通道的属性
        self.assertTrue(hasattr(model, 'low_freq_embedding'), "应该有 low_freq_embedding")
        self.assertTrue(hasattr(model, 'high_freq_embedding'), "应该有 high_freq_embedding")
        self.assertTrue(hasattr(model, 'gate'), "应该有 gate")
        
        # 测试前向传播
        B, N, T = 4, 7, 512
        x = torch.randn(B, N, T).to(self.device)
        output, n_vars = model(x)
        
        # 验证输出维度
        expected_num_patches = (T + 8 - 16) // 8 + 1  # (T + stride - patch_len) // stride + 1
        self.assertEqual(output.shape[0], B * N, f"Batch 维度应为 {B*N}")
        self.assertEqual(output.shape[2], 32, "d_model 应为 32")
        self.assertEqual(n_vars, N, f"n_vars 应为 {N}")
        
        print(f"✅ 输入: ({B}, {N}, {T})")
        print(f"✅ 输出: {output.shape}")
        print(f"✅ n_vars: {n_vars}")
        print("✅ level=1 双通道模式测试通过!")
    
    def test_level2_pyramid_fusion_mode(self):
        """测试 level=2 时使用金字塔融合模式"""
        print("\n" + "="*60)
        print("测试 2: level=2 金字塔融合模式")
        print("="*60)
        
        model = self.WISTPatchEmbedding(
            d_model=64,
            patch_len=16,
            stride=8,
            dropout=0.1,
            wavelet_type='db4',
            wavelet_level=2,
            hf_dropout=0.5,
            gate_bias_init=2.0,
            use_soft_threshold=True,
            use_causal_conv=True,
            pyramid_fusion=True,
            mf_dropout=0.3
        ).to(self.device)
        
        # 验证: level=2 时应该启用金字塔融合
        self.assertTrue(model.pyramid_fusion, "level=2 时应启用金字塔融合")
        
        # 验证: 频段数量 = level + 1 = 3
        self.assertEqual(model.num_bands, 3, "level=2 时频段数量应为 3")
        
        # 验证: 应该有 3 个 band_embeddings
        self.assertEqual(len(model.band_embeddings), 3, "应该有 3 个 band_embeddings")
        
        # 验证: 应该有 2 个门控层 (num_bands - 1)
        self.assertEqual(len(model.gate_layers), 2, "应该有 2 个门控层")
        
        # 验证: 应该有 3 个 dropout (1 个 Identity + 2 个 Dropout)
        self.assertEqual(len(model.band_dropouts), 3, "应该有 3 个 band_dropouts")
        
        # 测试前向传播
        B, N, T = 4, 7, 512
        x = torch.randn(B, N, T).to(self.device)
        output, n_vars = model(x)
        
        # 验证输出维度
        self.assertEqual(output.shape[0], B * N, f"Batch 维度应为 {B*N}")
        self.assertEqual(output.shape[2], 64, "d_model 应为 64")
        self.assertEqual(n_vars, N, f"n_vars 应为 {N}")
        
        print(f"✅ 输入: ({B}, {N}, {T})")
        print(f"✅ 输出: {output.shape}")
        print(f"✅ 频段数量: {model.num_bands}")
        print(f"✅ 门控层数量: {len(model.gate_layers)}")
        print("✅ level=2 金字塔融合模式测试通过!")
    
    def test_level3_pyramid_fusion_mode(self):
        """测试 level=3 时使用金字塔融合模式"""
        print("\n" + "="*60)
        print("测试 3: level=3 金字塔融合模式")
        print("="*60)
        
        model = self.WISTPatchEmbedding(
            d_model=64,
            patch_len=16,
            stride=8,
            dropout=0.1,
            wavelet_type='db4',
            wavelet_level=3,
            hf_dropout=0.5,
            gate_bias_init=2.0,
            use_soft_threshold=True,
            use_causal_conv=True,
            pyramid_fusion=True,
            mf_dropout=0.2
        ).to(self.device)
        
        # 验证: 频段数量 = level + 1 = 4
        self.assertEqual(model.num_bands, 4, "level=3 时频段数量应为 4")
        
        # 验证: 应该有 3 个门控层 (num_bands - 1)
        self.assertEqual(len(model.gate_layers), 3, "应该有 3 个门控层")
        
        # 测试前向传播
        B, N, T = 4, 7, 512
        x = torch.randn(B, N, T).to(self.device)
        output, n_vars = model(x)
        
        print(f"✅ 输入: ({B}, {N}, {T})")
        print(f"✅ 输出: {output.shape}")
        print(f"✅ 频段数量: {model.num_bands} (cA + cD_3 + cD_2 + cD_1)")
        print(f"✅ 门控层数量: {len(model.gate_layers)}")
        print("✅ level=3 金字塔融合模式测试通过!")
    
    def test_gradient_flow(self):
        """测试梯度能正常回传"""
        print("\n" + "="*60)
        print("测试 4: 梯度回传")
        print("="*60)
        
        model = self.WISTPatchEmbedding(
            d_model=32,
            patch_len=16,
            stride=8,
            dropout=0.0,  # 关闭 dropout 以便测试梯度
            wavelet_type='db4',
            wavelet_level=2,
            hf_dropout=0.0,
            gate_bias_init=0.0,
            use_soft_threshold=True,
            use_causal_conv=True,
            pyramid_fusion=True,
            mf_dropout=0.0
        ).to(self.device)
        
        # 前向传播 - 使用叶子张量
        B, N, T = 2, 3, 256
        x = torch.randn(B, N, T, device=self.device, requires_grad=True)
        output, n_vars = model(x)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 验证输入梯度存在 (x 是叶子张量，所以 grad 应该存在)
        self.assertIsNotNone(x.grad, "输入应该有梯度")
        self.assertTrue(torch.any(x.grad != 0), "输入梯度不应全为零")
        
        # 验证模型参数有梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break
        self.assertTrue(has_grad, "模型参数应该有非零梯度")
        
        print(f"✅ 输入梯度形状: {x.grad.shape}")
        print(f"✅ 输入梯度范数: {x.grad.norm().item():.6f}")
        print("✅ 梯度回传测试通过!")
    
    def test_pyramid_fusion_disabled(self):
        """测试禁用金字塔融合时 level>=2 的行为"""
        print("\n" + "="*60)
        print("测试 5: 禁用金字塔融合 (level=2)")
        print("="*60)
        
        model = self.WISTPatchEmbedding(
            d_model=32,
            patch_len=16,
            stride=8,
            dropout=0.1,
            wavelet_type='db4',
            wavelet_level=2,
            hf_dropout=0.5,
            gate_bias_init=2.0,
            use_soft_threshold=True,
            use_causal_conv=True,
            pyramid_fusion=False,  # 显式禁用
            mf_dropout=0.3
        ).to(self.device)
        
        # 验证: 即使 level=2，禁用后也应该用双通道模式
        self.assertFalse(model.pyramid_fusion, "显式禁用时不应启用金字塔融合")
        self.assertTrue(hasattr(model, 'low_freq_embedding'), "应该有 low_freq_embedding")
        
        # 测试前向传播
        B, N, T = 4, 7, 512
        x = torch.randn(B, N, T).to(self.device)
        output, n_vars = model(x)
        
        print(f"✅ 输入: ({B}, {N}, {T})")
        print(f"✅ 输出: {output.shape}")
        print("✅ 禁用金字塔融合测试通过!")
    
    def test_dropout_interpolation(self):
        """测试 Dropout 率的线性插值"""
        print("\n" + "="*60)
        print("测试 6: Dropout 率线性插值")
        print("="*60)
        
        mf_dropout = 0.2
        hf_dropout = 0.6
        
        model = self.WISTPatchEmbedding(
            d_model=32,
            patch_len=16,
            stride=8,
            dropout=0.1,
            wavelet_type='db4',
            wavelet_level=3,  # 4 个频段: cA, cD_3, cD_2, cD_1
            hf_dropout=hf_dropout,
            gate_bias_init=2.0,
            use_soft_threshold=False,
            use_causal_conv=False,
            pyramid_fusion=True,
            mf_dropout=mf_dropout
        ).to(self.device)
        
        # band_dropouts[0] = Identity (cA)
        # band_dropouts[1] = Dropout(mf_dropout) (cD_3, 最低的高频)
        # band_dropouts[2] = Dropout(中间值) (cD_2)
        # band_dropouts[3] = Dropout(hf_dropout) (cD_1, 最高频)
        
        # 验证 cA 是 Identity
        self.assertIsInstance(model.band_dropouts[0], nn.Identity, "cA 应该是 Identity")
        
        # 验证 cD_3 (index=1) 的 dropout 率接近 mf_dropout
        self.assertIsInstance(model.band_dropouts[1], nn.Dropout)
        self.assertAlmostEqual(model.band_dropouts[1].p, mf_dropout, places=2,
                               msg=f"cD_3 的 dropout 应接近 {mf_dropout}")
        
        # 验证 cD_1 (index=3) 的 dropout 率接近 hf_dropout
        self.assertIsInstance(model.band_dropouts[3], nn.Dropout)
        self.assertAlmostEqual(model.band_dropouts[3].p, hf_dropout, places=2,
                               msg=f"cD_1 的 dropout 应接近 {hf_dropout}")
        
        print(f"✅ cA (index=0): Identity")
        print(f"✅ cD_3 (index=1): Dropout(p={model.band_dropouts[1].p:.2f})")
        print(f"✅ cD_2 (index=2): Dropout(p={model.band_dropouts[2].p:.2f})")
        print(f"✅ cD_1 (index=3): Dropout(p={model.band_dropouts[3].p:.2f})")
        print("✅ Dropout 率线性插值测试通过!")
    
    def test_output_consistency(self):
        """测试 eval 模式下输出的一致性"""
        print("\n" + "="*60)
        print("测试 7: eval 模式输出一致性")
        print("="*60)
        
        model = self.WISTPatchEmbedding(
            d_model=32,
            patch_len=16,
            stride=8,
            dropout=0.1,
            wavelet_type='db4',
            wavelet_level=2,
            hf_dropout=0.5,
            gate_bias_init=2.0,
            use_soft_threshold=True,
            use_causal_conv=True,
            pyramid_fusion=True,
            mf_dropout=0.3
        ).to(self.device)
        
        model.eval()  # 切换到评估模式
        
        B, N, T = 2, 3, 256
        x = torch.randn(B, N, T).to(self.device)
        
        with torch.no_grad():
            output1, _ = model(x)
            output2, _ = model(x)
        
        # eval 模式下，相同输入应该产生相同输出
        self.assertTrue(torch.allclose(output1, output2), 
                        "eval 模式下相同输入应产生相同输出")
        
        print(f"✅ 输出差异: {(output1 - output2).abs().max().item():.10f}")
        print("✅ eval 模式输出一致性测试通过!")


class TestCausalSWTIntegration(unittest.TestCase):
    """测试 CausalSWT 与金字塔融合的集成"""
    
    @classmethod
    def setUpClass(cls):
        from layers.CausalWavelet import CausalSWT
        cls.CausalSWT = CausalSWT
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_swt_output_shape(self):
        """测试 CausalSWT 输出形状"""
        print("\n" + "="*60)
        print("测试 8: CausalSWT 输出形状")
        print("="*60)
        
        for level in [1, 2, 3]:
            swt = self.CausalSWT(wavelet='db4', level=level)
            
            B, N, T = 4, 7, 512
            # CausalSWT 的滤波器在 CPU 上，所以输入也用 CPU
            x = torch.randn(B, N, T)
            coeffs = swt(x)
            
            # 输出形状应为 (B, N, T, level+1)
            expected_shape = (B, N, T, level + 1)
            self.assertEqual(coeffs.shape, expected_shape,
                             f"level={level} 时输出形状应为 {expected_shape}")
            
            print(f"✅ level={level}: 输入 ({B}, {N}, {T}) -> 输出 {coeffs.shape}")
        
        print("✅ CausalSWT 输出形状测试通过!")


if __name__ == '__main__':
    print("=" * 70)
    print("WISTPatchEmbedding 金字塔融合功能测试")
    print("=" * 70)
    
    # 运行测试
    unittest.main(verbosity=2)
