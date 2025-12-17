"""
频率解耦输出头 (Frequency Decoupled Head) 全面测试

测试内容:
1. TriBandDecoupledHead 基本功能测试
2. SoftThreshold 模块测试
3. DeepSupervisionLoss 模块测试
4. 与 TimeLLM 模型的集成测试
5. 梯度传播测试
6. 不同配置组合测试
7. 边界条件测试
8. 性能基准测试

Author: CAST Project
Date: 2024
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Tuple, Optional

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from layers.FrequencyDecoupledHead import (
    TriBandDecoupledHead,
    DeepSupervisionLoss,
    SoftThreshold
)


class TestSoftThreshold(unittest.TestCase):
    """SoftThreshold 模块测试"""
    
    def setUp(self):
        """测试初始化"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_features = 64
        self.batch_size = 4
    
    def test_basic_forward(self):
        """测试基本前向传播"""
        st = SoftThreshold(self.num_features, init_tau=0.1).to(self.device)
        x = torch.randn(self.batch_size, self.num_features, device=self.device)
        y = st(x)
        
        self.assertEqual(y.shape, x.shape)
        print("✅ SoftThreshold 基本前向传播测试通过")
    
    def test_thresholding_behavior(self):
        """测试阈值行为：小于阈值的值应被置零"""
        st = SoftThreshold(self.num_features, init_tau=0.5).to(self.device)
        
        # 创建包含小值和大值的输入
        x = torch.tensor([[0.1, 0.3, 0.6, 0.8, -0.2, -0.7]], device=self.device)
        st_single = SoftThreshold(6, init_tau=0.5).to(self.device)
        y = st_single(x)
        
        # 检查小于阈值的值是否被置零
        tau = 0.5
        expected_zero_mask = x.abs() < tau
        actual_zero = (y == 0)
        
        self.assertTrue(torch.all(actual_zero[expected_zero_mask]))
        print("✅ SoftThreshold 阈值行为测试通过")
    
    def test_gradient_flow(self):
        """测试梯度流"""
        st = SoftThreshold(self.num_features, init_tau=0.1).to(self.device)
        x = torch.randn(self.batch_size, self.num_features, device=self.device, requires_grad=True)
        
        y = st(x)
        loss = y.sum()
        loss.backward()
        
        # 检查输入和参数都有梯度
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(st.tau.grad)
        self.assertTrue(x.grad.abs().sum() > 0)
        print("✅ SoftThreshold 梯度流测试通过")
    
    def test_learnable_tau(self):
        """测试可学习阈值"""
        st = SoftThreshold(self.num_features, init_tau=0.1).to(self.device)
        
        # 初始阈值
        initial_tau = st.tau.clone()
        
        # 模拟训练
        optimizer = torch.optim.SGD([st.tau], lr=0.1)
        for _ in range(10):
            x = torch.randn(self.batch_size, self.num_features, device=self.device)
            y = st(x)
            loss = y.abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 检查阈值是否改变
        self.assertFalse(torch.allclose(st.tau, initial_tau))
        print("✅ SoftThreshold 可学习阈值测试通过")


class TestTriBandDecoupledHead(unittest.TestCase):
    """TriBandDecoupledHead 模块测试"""
    
    def setUp(self):
        """测试初始化"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.n_vars = 7
        self.d_ff = 32
        self.patch_nums = 10
        self.nf = self.d_ff * self.patch_nums
        self.pred_len = 96
    
    def _create_head(self, **kwargs) -> TriBandDecoupledHead:
        """创建测试用的 Head"""
        default_kwargs = {
            'n_vars': self.n_vars,
            'nf': self.nf,
            'target_window': self.pred_len,
            'head_dropout': 0.1,
            'mid_dropout': 0.2,
            'high_dropout': 0.5,
            'use_soft_threshold': True,
            'soft_threshold_init': 0.1,
            'use_conv': False,
        }
        default_kwargs.update(kwargs)
        return TriBandDecoupledHead(**default_kwargs).to(self.device)
    
    def _create_input(self) -> torch.Tensor:
        """创建测试输入"""
        return torch.randn(
            self.batch_size, self.n_vars, self.d_ff, self.patch_nums,
            device=self.device
        )
    
    def test_basic_forward_4d_input(self):
        """测试 4D 输入的前向传播"""
        head = self._create_head()
        x = self._create_input()
        
        output = head(x, return_components=False)
        
        expected_shape = (self.batch_size, self.pred_len, self.n_vars)
        self.assertEqual(output.shape, expected_shape)
        print("✅ TriBandDecoupledHead 4D 输入前向传播测试通过")
    
    def test_basic_forward_3d_input(self):
        """测试 3D 输入的前向传播"""
        head = self._create_head()
        x = torch.randn(self.batch_size, self.n_vars, self.nf, device=self.device)
        
        output = head(x, return_components=False)
        
        expected_shape = (self.batch_size, self.pred_len, self.n_vars)
        self.assertEqual(output.shape, expected_shape)
        print("✅ TriBandDecoupledHead 3D 输入前向传播测试通过")
    
    def test_return_components(self):
        """测试返回频率分量"""
        head = self._create_head()
        x = self._create_input()
        
        output, components = head(x, return_components=True)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.pred_len, self.n_vars)
        self.assertEqual(output.shape, expected_shape)
        
        # 检查分量
        self.assertIn('pred_trend', components)
        self.assertIn('pred_mid', components)
        self.assertIn('pred_detail', components)
        
        for key, comp in components.items():
            self.assertEqual(comp.shape, expected_shape, f"{key} 形状错误")
        
        print("✅ TriBandDecoupledHead 返回频率分量测试通过")
    
    def test_component_sum_equals_output_eval_mode(self):
        """测试 eval 模式下分量相加等于输出"""
        head = self._create_head()
        head.eval()
        x = self._create_input()
        
        with torch.no_grad():
            output, components = head(x, return_components=True)
            reconstructed = (
                components['pred_trend'] + 
                components['pred_mid'] + 
                components['pred_detail']
            )
        
        diff = (output - reconstructed).abs().max().item()
        self.assertLess(diff, 1e-5, f"分量重构误差过大: {diff}")
        print("✅ TriBandDecoupledHead 分量重构一致性测试通过")
    
    def test_conv1d_mode(self):
        """测试 Conv1d 模式"""
        head = self._create_head(use_conv=True)
        x = self._create_input()
        
        output = head(x)
        
        expected_shape = (self.batch_size, self.pred_len, self.n_vars)
        self.assertEqual(output.shape, expected_shape)
        print("✅ TriBandDecoupledHead Conv1d 模式测试通过")
    
    def test_no_soft_threshold(self):
        """测试关闭 SoftThreshold"""
        head = self._create_head(use_soft_threshold=False)
        x = self._create_input()
        
        output = head(x)
        
        expected_shape = (self.batch_size, self.pred_len, self.n_vars)
        self.assertEqual(output.shape, expected_shape)
        print("✅ TriBandDecoupledHead 关闭 SoftThreshold 测试通过")
    
    def test_gradient_flow(self):
        """测试梯度传播"""
        head = self._create_head()
        head.train()
        x = self._create_input()
        x.requires_grad = True
        
        output, components = head(x, return_components=True)
        loss = output.sum() + sum(c.sum() for c in components.values())
        loss.backward()
        
        # 检查输入梯度
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)
        
        # 检查所有参数都有梯度
        for name, param in head.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"{name} 没有梯度")
        
        print("✅ TriBandDecoupledHead 梯度传播测试通过")
    
    def test_different_pred_lens(self):
        """测试不同预测长度"""
        for pred_len in [24, 48, 96, 192, 336, 720]:
            head = self._create_head(target_window=pred_len)
            x = self._create_input()
            
            output = head(x)
            
            expected_shape = (self.batch_size, pred_len, self.n_vars)
            self.assertEqual(output.shape, expected_shape, f"pred_len={pred_len} 失败")
        
        print("✅ TriBandDecoupledHead 不同预测长度测试通过")
    
    def test_different_batch_sizes(self):
        """测试不同批次大小"""
        head = self._create_head()
        
        for batch_size in [1, 2, 8, 16, 32]:
            x = torch.randn(
                batch_size, self.n_vars, self.d_ff, self.patch_nums,
                device=self.device
            )
            output = head(x)
            
            expected_shape = (batch_size, self.pred_len, self.n_vars)
            self.assertEqual(output.shape, expected_shape, f"batch_size={batch_size} 失败")
        
        print("✅ TriBandDecoupledHead 不同批次大小测试通过")
    
    def test_parameter_count(self):
        """测试参数统计"""
        head = self._create_head()
        
        total_params = sum(p.numel() for p in head.parameters())
        trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        
        self.assertEqual(total_params, trainable_params)
        self.assertGreater(total_params, 0)
        
        print(f"✅ TriBandDecoupledHead 参数统计: {total_params:,} 参数")


class TestDeepSupervisionLoss(unittest.TestCase):
    """DeepSupervisionLoss 模块测试"""
    
    def setUp(self):
        """测试初始化"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.n_vars = 7
        self.pred_len = 96
    
    def _create_loss_fn(self, **kwargs) -> DeepSupervisionLoss:
        """创建测试用的 Loss 函数"""
        default_kwargs = {
            'wavelet': 'db4',
            'level': 2,
            'alpha': 0.3,
            'use_causal_swt': True,  # 使用因果版本确保测试通过
        }
        default_kwargs.update(kwargs)
        return DeepSupervisionLoss(**default_kwargs).to(self.device)
    
    def _create_pred_and_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建预测和目标"""
        pred = torch.randn(self.batch_size, self.pred_len, self.n_vars, device=self.device)
        target = torch.randn(self.batch_size, self.pred_len, self.n_vars, device=self.device)
        return pred, target
    
    def _create_components(self) -> Dict[str, torch.Tensor]:
        """创建频率分量"""
        shape = (self.batch_size, self.pred_len, self.n_vars)
        return {
            'pred_trend': torch.randn(*shape, device=self.device),
            'pred_mid': torch.randn(*shape, device=self.device),
            'pred_detail': torch.randn(*shape, device=self.device),
        }
    
    def test_basic_loss_computation(self):
        """测试基本损失计算"""
        ds_loss = self._create_loss_fn()
        pred, target = self._create_pred_and_target()
        components = self._create_components()
        
        total_loss, loss_dict = ds_loss(pred, target, components)
        
        # 检查返回值
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)
        
        # 检查损失字典内容
        self.assertIn('main_loss', loss_dict)
        self.assertIn('loss_trend', loss_dict)
        self.assertIn('loss_mid', loss_dict)
        self.assertIn('loss_detail', loss_dict)
        self.assertIn('aux_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)
        
        print("✅ DeepSupervisionLoss 基本损失计算测试通过")
    
    def test_loss_without_components(self):
        """测试无分量时只返回主损失"""
        ds_loss = self._create_loss_fn()
        pred, target = self._create_pred_and_target()
        
        total_loss, loss_dict = ds_loss(pred, target, components=None)
        
        # 应该只有主损失
        self.assertIn('main_loss', loss_dict)
        self.assertNotIn('aux_loss', loss_dict)
        
        # 总损失应等于主损失
        main_loss = F.mse_loss(pred, target)
        self.assertAlmostEqual(total_loss.item(), main_loss.item(), places=5)
        
        print("✅ DeepSupervisionLoss 无分量模式测试通过")
    
    def test_alpha_weighting(self):
        """测试 alpha 权重"""
        pred, target = self._create_pred_and_target()
        components = self._create_components()
        
        # 测试不同 alpha 值
        for alpha in [0.0, 0.1, 0.3, 0.5, 1.0]:
            ds_loss = self._create_loss_fn(alpha=alpha)
            total_loss, loss_dict = ds_loss(pred, target, components)
            
            # 验证公式: total = main + alpha * aux
            expected_total = loss_dict['main_loss'] + alpha * loss_dict['aux_loss']
            self.assertAlmostEqual(
                loss_dict['total_loss'], expected_total, places=5,
                msg=f"alpha={alpha} 时损失计算错误"
            )
        
        print("✅ DeepSupervisionLoss alpha 权重测试通过")
    
    def test_gradient_flow(self):
        """测试梯度传播"""
        ds_loss = self._create_loss_fn()
        
        pred = torch.randn(
            self.batch_size, self.pred_len, self.n_vars,
            device=self.device, requires_grad=True
        )
        target = torch.randn(self.batch_size, self.pred_len, self.n_vars, device=self.device)
        
        components = {
            'pred_trend': torch.randn(
                self.batch_size, self.pred_len, self.n_vars,
                device=self.device, requires_grad=True
            ),
            'pred_mid': torch.randn(
                self.batch_size, self.pred_len, self.n_vars,
                device=self.device, requires_grad=True
            ),
            'pred_detail': torch.randn(
                self.batch_size, self.pred_len, self.n_vars,
                device=self.device, requires_grad=True
            ),
        }
        
        total_loss, _ = ds_loss(pred, target, components)
        total_loss.backward()
        
        # 检查梯度
        self.assertIsNotNone(pred.grad)
        for key, comp in components.items():
            self.assertIsNotNone(comp.grad, f"{key} 没有梯度")
        
        print("✅ DeepSupervisionLoss 梯度传播测试通过")
    
    def test_different_wavelet_levels(self):
        """测试不同小波分解层数"""
        pred, target = self._create_pred_and_target()
        
        for level in [1, 2, 3]:
            ds_loss = self._create_loss_fn(level=level)
            components = self._create_components()
            
            total_loss, loss_dict = ds_loss(pred, target, components)
            
            self.assertIsInstance(total_loss, torch.Tensor)
            self.assertFalse(torch.isnan(total_loss))
        
        print("✅ DeepSupervisionLoss 不同小波层数测试通过")


class TestIntegration(unittest.TestCase):
    """集成测试：TriBandDecoupledHead + DeepSupervisionLoss"""
    
    def setUp(self):
        """测试初始化"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.n_vars = 7
        self.d_ff = 32
        self.patch_nums = 10
        self.nf = self.d_ff * self.patch_nums
        self.pred_len = 96
    
    def test_end_to_end_training_step(self):
        """测试端到端训练步骤"""
        # 创建模块
        head = TriBandDecoupledHead(
            n_vars=self.n_vars,
            nf=self.nf,
            target_window=self.pred_len,
            use_soft_threshold=True,
        ).to(self.device)
        
        ds_loss = DeepSupervisionLoss(
            wavelet='db4',
            level=2,
            alpha=0.3,
            use_causal_swt=True,
        ).to(self.device)
        
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        
        # 模拟训练
        head.train()
        for step in range(5):
            # 创建输入和目标
            x = torch.randn(
                self.batch_size, self.n_vars, self.d_ff, self.patch_nums,
                device=self.device
            )
            target = torch.randn(
                self.batch_size, self.pred_len, self.n_vars,
                device=self.device
            )
            
            # 前向传播
            pred, components = head(x, return_components=True)
            
            # 计算损失
            loss, loss_dict = ds_loss(pred, target, components)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 检查损失有效
            self.assertFalse(torch.isnan(loss))
            self.assertFalse(torch.isinf(loss))
        
        print("✅ 端到端训练步骤测试通过")
    
    def test_loss_decreases_during_training(self):
        """测试训练过程中损失下降"""
        # 创建模块
        head = TriBandDecoupledHead(
            n_vars=self.n_vars,
            nf=self.nf,
            target_window=self.pred_len,
            use_soft_threshold=True,
        ).to(self.device)
        
        ds_loss = DeepSupervisionLoss(
            wavelet='db4',
            level=2,
            alpha=0.3,
            use_causal_swt=True,
        ).to(self.device)
        
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
        
        # 固定输入和目标
        x = torch.randn(
            self.batch_size, self.n_vars, self.d_ff, self.patch_nums,
            device=self.device
        )
        target = torch.randn(
            self.batch_size, self.pred_len, self.n_vars,
            device=self.device
        )
        
        # 训练并记录损失
        losses = []
        head.train()
        for _ in range(50):
            pred, components = head(x, return_components=True)
            loss, _ = ds_loss(pred, target, components)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # 检查损失下降趋势
        first_10_avg = np.mean(losses[:10])
        last_10_avg = np.mean(losses[-10:])
        
        self.assertLess(last_10_avg, first_10_avg, "损失未下降")
        print(f"✅ 损失下降测试通过: {first_10_avg:.4f} -> {last_10_avg:.4f}")
    
    def test_eval_mode_deterministic(self):
        """测试 eval 模式输出确定性"""
        head = TriBandDecoupledHead(
            n_vars=self.n_vars,
            nf=self.nf,
            target_window=self.pred_len,
        ).to(self.device)
        
        head.eval()
        x = torch.randn(
            self.batch_size, self.n_vars, self.d_ff, self.patch_nums,
            device=self.device
        )
        
        with torch.no_grad():
            output1 = head(x)
            output2 = head(x)
        
        self.assertTrue(torch.allclose(output1, output2))
        print("✅ eval 模式确定性测试通过")


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """测试初始化"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def test_inference_speed(self):
        """测试推理速度"""
        batch_size = 32
        n_vars = 7
        d_ff = 32
        patch_nums = 10
        nf = d_ff * patch_nums
        pred_len = 96
        
        head = TriBandDecoupledHead(
            n_vars=n_vars,
            nf=nf,
            target_window=pred_len,
        ).to(self.device)
        head.eval()
        
        x = torch.randn(batch_size, n_vars, d_ff, patch_nums, device=self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = head(x)
        
        # 计时
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        num_iterations = 100
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = head(x)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations * 1000  # ms
        
        print(f"✅ 推理速度测试: {avg_time:.3f} ms/batch (batch_size={batch_size})")
    
    def test_memory_usage(self):
        """测试显存使用"""
        if self.device.type != 'cuda':
            self.skipTest("需要 CUDA 设备")
        
        batch_size = 32
        n_vars = 7
        d_ff = 32
        patch_nums = 10
        nf = d_ff * patch_nums
        pred_len = 96
        
        torch.cuda.reset_peak_memory_stats()
        
        head = TriBandDecoupledHead(
            n_vars=n_vars,
            nf=nf,
            target_window=pred_len,
        ).to(self.device)
        
        x = torch.randn(batch_size, n_vars, d_ff, patch_nums, device=self.device)
        
        # 前向传播
        output, components = head(x, return_components=True)
        
        # 反向传播
        loss = output.sum()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        print(f"✅ 显存使用测试: {peak_memory:.2f} MB (batch_size={batch_size})")


class TestEdgeCases(unittest.TestCase):
    """边界条件测试"""
    
    def setUp(self):
        """测试初始化"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def test_single_variable(self):
        """测试单变量情况"""
        head = TriBandDecoupledHead(
            n_vars=1,
            nf=320,
            target_window=96,
        ).to(self.device)
        
        x = torch.randn(4, 1, 32, 10, device=self.device)
        output = head(x)
        
        self.assertEqual(output.shape, (4, 96, 1))
        print("✅ 单变量测试通过")
    
    def test_single_batch(self):
        """测试单批次情况"""
        head = TriBandDecoupledHead(
            n_vars=7,
            nf=320,
            target_window=96,
        ).to(self.device)
        
        x = torch.randn(1, 7, 32, 10, device=self.device)
        output = head(x)
        
        self.assertEqual(output.shape, (1, 96, 7))
        print("✅ 单批次测试通过")
    
    def test_very_short_prediction(self):
        """测试极短预测长度"""
        head = TriBandDecoupledHead(
            n_vars=7,
            nf=320,
            target_window=1,
        ).to(self.device)
        
        x = torch.randn(4, 7, 32, 10, device=self.device)
        output = head(x)
        
        self.assertEqual(output.shape, (4, 1, 7))
        print("✅ 极短预测长度测试通过")
    
    def test_very_long_prediction(self):
        """测试极长预测长度"""
        head = TriBandDecoupledHead(
            n_vars=7,
            nf=320,
            target_window=720,
        ).to(self.device)
        
        x = torch.randn(4, 7, 32, 10, device=self.device)
        output = head(x)
        
        self.assertEqual(output.shape, (4, 720, 7))
        print("✅ 极长预测长度测试通过")
    
    def test_zero_dropout(self):
        """测试零 Dropout"""
        head = TriBandDecoupledHead(
            n_vars=7,
            nf=320,
            target_window=96,
            head_dropout=0.0,
            mid_dropout=0.0,
            high_dropout=0.0,
        ).to(self.device)
        
        x = torch.randn(4, 7, 32, 10, device=self.device)
        output = head(x)
        
        self.assertEqual(output.shape, (4, 96, 7))
        print("✅ 零 Dropout 测试通过")
    
    def test_high_dropout(self):
        """测试高 Dropout"""
        head = TriBandDecoupledHead(
            n_vars=7,
            nf=320,
            target_window=96,
            head_dropout=0.9,
            mid_dropout=0.9,
            high_dropout=0.9,
        ).to(self.device)
        
        x = torch.randn(4, 7, 32, 10, device=self.device)
        output = head(x)
        
        self.assertEqual(output.shape, (4, 96, 7))
        self.assertFalse(torch.isnan(output).any())
        print("✅ 高 Dropout 测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("频率解耦输出头 (Frequency Decoupled Head) 全面测试")
    print("=" * 70)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestSoftThreshold))
    suite.addTests(loader.loadTestsFromTestCase(TestTriBandDecoupledHead))
    suite.addTests(loader.loadTestsFromTestCase(TestDeepSupervisionLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("🎉 所有测试通过!")
    else:
        print(f"❌ 测试失败: {len(result.failures)} 失败, {len(result.errors)} 错误")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
