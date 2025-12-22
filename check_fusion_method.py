#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查 fusion_method 参数传递的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from models.TimeLLM import Model

# 创建参数解析器（与 run_main.py 相同）
parser = argparse.ArgumentParser(description='Check fusion method')
parser.add_argument('--use_dual_prototypes', type=int, default=1)
parser.add_argument('--dual_proto_fusion_method', type=str, default='mean',
                    choices=['mean', 'weighted', 'adaptive_gate', 'interleave', 'channel_concat'])

# 添加必要的参数（避免报错）
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--llm_model', type=str, default='GPT2')
parser.add_argument('--llm_dim', type=int, default=768)
parser.add_argument('--llm_layers', type=int, default=2)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--wavelet_mode', type=str, default='wist')
parser.add_argument('--wavelet_type', type=str, default='haar')
parser.add_argument('--wavelet_level', type=int, default=2)
parser.add_argument('--use_semantic_filtered_mapping', type=int, default=1)
parser.add_argument('--dual_proto_trend_seed_words', type=int, default=1000)
parser.add_argument('--dual_proto_detail_seed_words', type=int, default=1000)
parser.add_argument('--dual_proto_seed_semantic_filter', type=int, default=1)
parser.add_argument('--dual_proto_mlp_hidden_dim', type=int, default=2048)
parser.add_argument('--dual_proto_mlp_dropout', type=float, default=0.1)
parser.add_argument('--dual_proto_gate_bias_init', type=float, default=0.0)
parser.add_argument('--use_cwpr', type=int, default=0)
parser.add_argument('--use_dual_scale_head', type=int, default=0)
parser.add_argument('--use_freq_decoupled_head', type=int, default=0)
parser.add_argument('--prompt_domain', type=int, default=0)
parser.add_argument('--content', type=str, default='Test')

args = parser.parse_args()

print("=" * 70)
print("检查 fusion_method 参数传递")
print("=" * 70)
print(f"\n命令行参数:")
print(f"  --use_dual_prototypes: {args.use_dual_prototypes}")
print(f"  --dual_proto_fusion_method: {args.dual_proto_fusion_method}")

print(f"\nargs 对象属性:")
print(f"  args.dual_proto_fusion_method = '{args.dual_proto_fusion_method}'")
print(f"  hasattr(args, 'dual_proto_fusion_method') = {hasattr(args, 'dual_proto_fusion_method')}")

print(f"\n尝试创建模型（会打印调试信息）...")
try:
    model = Model(args)
    print(f"\n✅ 模型创建成功")
    print(f"   model.fusion_method = {getattr(model, 'fusion_method', 'NOT_FOUND')}")
    if hasattr(model, 'reprogramming_layer') and hasattr(model.reprogramming_layer, 'fusion_method'):
        print(f"   model.reprogramming_layer.fusion_method = {model.reprogramming_layer.fusion_method}")
except Exception as e:
    print(f"\n❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()

